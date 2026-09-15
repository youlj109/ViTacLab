#!/usr/bin/env python3
"""Replay a relative depth gain with fixed nut optics and saved marker motion."""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher
from sweep_xense_force_depth_gain import CASES, _load_rgb, _label_above as _label, _clean_markers, _align_frame_background


def save(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path('logs/xense_flat_reference_refined_20260915'))
    parser.add_argument('--output', type=Path, default=Path('logs/xense_depth_gain110_20260915'))
    parser.add_argument('--relative-gain', type=float, default=1.1)
    parser.add_argument('--fit-to-real', action='store_true')
    parser.add_argument('--baseline-gain', type=float, default=1.0)
    parser.add_argument('--rgb-response-gains', type=float, nargs='+', default=[1.0])
    parser.add_argument('--selection', choices=['train','all'], default='train')
    parser.add_argument('--candidate-k-refs', type=float, nargs='+', default=None,
                        help='Fit effective stiffness in N/m, holding target-depth-gain fixed.')
    parser.add_argument('--source-k-ref', type=float, default=1840.)
    parser.add_argument('--source-depth-gain', type=float, default=1.)
    parser.add_argument('--target-depth-gain', type=float, default=1.)
    parser.add_argument('--candidate-depth-gains', type=float, nargs='+', default=None,
                        help='Joint 2D grid with candidate-k-refs; equivalent ratios share a render.')
    parser.add_argument('--baseline-k-ref', type=float, default=None)
    parser.add_argument('--baseline-depth-gain', type=float, default=1.)
    parser.add_argument('--candidate-gains', type=float, nargs='+',
                        default=[.8,1.,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.,2.2,2.5,3.])
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    joint_pairs={}
    if args.candidate_depth_gains and not args.candidate_k_refs:
        parser.error('--candidate-depth-gains requires --candidate-k-refs')
    if args.candidate_depth_gains and args.rgb_response_gains != [1.0]:
        parser.error('Joint k_ref/depth_gain scan holds --rgb-response-gains at 1.0')
    for value in [args.source_k_ref,args.source_depth_gain,args.baseline_depth_gain]:
        if not np.isfinite(value) or value<=0:
            parser.error('Source stiffness and depth gains must be finite and positive')
    if args.baseline_k_ref is not None:
        if not np.isfinite(args.baseline_k_ref) or args.baseline_k_ref<=0:
            parser.error('--baseline-k-ref must be finite and positive')
        args.baseline_gain=args.source_k_ref/args.baseline_k_ref*args.baseline_depth_gain/args.source_depth_gain
    if not np.isfinite(args.relative_gain) or args.relative_gain <= 0:
        raise ValueError('relative-gain must be finite and positive')
    if args.candidate_k_refs is not None:
        depth_gains=args.candidate_depth_gains or [args.target_depth_gain]
        values=args.candidate_k_refs+depth_gains+[args.source_k_ref,args.source_depth_gain,args.target_depth_gain]
        if any(not np.isfinite(x) or x<=0 for x in values):
            raise ValueError('Stiffness and depth gains must be finite and positive')
        for k,d in itertools.product(args.candidate_k_refs,depth_gains):
            ratio=round(args.source_k_ref/k*d/args.source_depth_gain,12)
            joint_pairs.setdefault(ratio,[]).append(dict(k_ref=k,depth_gain=d))
        args.candidate_gains=list(joint_pairs)
    def effective_k(gain):
        return args.source_k_ref*args.target_depth_gain/(gain*args.source_depth_gain)
    app = AppLauncher(args).app
    from ViTacLab.assets.sensor.tacsl_sensor.gelsight_calibrated_cfg import validation_gelsight_render_cfg
    from ViTacLab.assets.sensor.tacsl_sensor.visuotactile_render import GelsightRender

    cfg = validation_gelsight_render_cfg(profile='advisor', flat_contact=True,
                                         enable_marker=True, marker_pattern='xense')
    renderer = GelsightRender(cfg, args.device)
    # Replay known FOTS displacements to isolate depth/optics from marker motion.
    marker = renderer._marker_sim
    renderer._marker_sim = None
    fit_report = None
    if args.fit_to_real:
        rest = marker.rest_xy.cpu().numpy()
        bg = _load_rgb(Path('data/calibration/tactile/advisor_processed/bg_clean.jpg'))
        heights = np.stack([np.load(args.input_root/'normal_force'/c/'vitacsim/tactile_height_corrected.npy') for c in CASES])
        height_batch = torch.as_tensor(heights,device=args.device)
        references, masks = [], []
        for case in CASES:
            real = _load_rgb(Path('data/calibration/tactile/real/normal_force')/case/'rgb.png')
            clean,mask = _clean_markers(real,rest)
            aligned,_ = _align_frame_background(clean,bg,marker_mask=mask,
                                                center_xy=(200.,350.),contact_radius_px=85.)
            references.append(aligned[240:460,90:310].astype(np.float32)-bg[240:460,90:310])
            masks.append(mask[240:460,90:310]==0)
        train,validation = [0,2,4],[1,3,5]
        trials=[]
        for gain,rgb_gain in itertools.product(sorted(set(args.candidate_gains+[args.baseline_gain])),args.rgb_response_gains):
            if gain <= 0 or not np.isfinite(gain) or rgb_gain <= 0 or not np.isfinite(rgb_gain):
                raise ValueError('Candidate gains must be finite and positive')
            renderer.cfg=cfg.replace(taxim_rgb_response_gain=rgb_gain)
            pred = renderer.render(height_batch*gain).cpu().numpy()
            cases=[]
            for i,case in enumerate(CASES):
                delta = pred[i,240:460,90:310].astype(np.float32)-bg[240:460,90:310]
                ref,sim = references[i][masks[i]],delta[masks[i]]
                rmse=float(np.sqrt(np.mean((sim-ref)**2)))
                ref_rms=float(np.sqrt(np.mean(ref**2)))
                cases.append(dict(case=case,rmse=rmse,normalized_rmse=rmse/max(ref_rms,1e-6),
                                  response_rms_ratio=float(np.sqrt(np.mean(sim**2)))/max(ref_rms,1e-6),
                                  rgb_correlation=float(np.corrcoef(ref.ravel(),sim.ravel())[0,1])))
            result=dict(gain=gain,rgb_response_gain=rgb_gain,cases=cases,
                        k_ref=effective_k(gain),depth_gain=args.target_depth_gain,
                        all=float(np.mean([c['normalized_rmse'] for c in cases])),
                        train=float(np.mean([cases[i]['normalized_rmse'] for i in train])),
                        validation=float(np.mean([cases[i]['normalized_rmse'] for i in validation])))
            trials.append(result)
            print('GAIN',gain,'train',result['train'],'validation',result['validation'],flush=True)
        baseline=next(x for x in trials if x['gain']==args.baseline_gain and x['rgb_response_gain']==1.)
        selected=min(trials,key=lambda x:x[args.selection])
        accepted=selected['validation'] < baseline['validation'] if args.selection=='train' else selected['all']<baseline['all']
        args.relative_gain=selected['gain'] if accepted else args.baseline_gain
        if args.candidate_depth_gains:
            pairs=joint_pairs.get(round(args.relative_gain,12),[])
            if pairs:
                pair=min(pairs,key=lambda p:(abs(np.log(p['k_ref']/(args.baseline_k_ref or 1350.))),
                                            abs(np.log(p['depth_gain']/args.baseline_depth_gain))))
                args.target_depth_gain=pair['depth_gain']
            elif args.baseline_k_ref is not None:
                args.target_depth_gain=args.baseline_depth_gain
        selected_rgb_gain=selected['rgb_response_gain'] if accepted else 1.
        renderer.cfg=cfg.replace(taxim_rgb_response_gain=selected_rgb_gain)
        fit_report=dict(train_cases=[CASES[i] for i in train],validation_cases=[CASES[i] for i in validation],
                        selected_candidate=selected['gain'],accepted=accepted,selected_gain=args.relative_gain,
                        selected_rgb_response_gain=selected_rgb_gain,selection=args.selection,
                        selected_k_ref=effective_k(args.relative_gain),target_depth_gain=args.target_depth_gain,
                        source_k_ref=args.source_k_ref,source_depth_gain=args.source_depth_gain,
                        trials=trials,note='Marker-excluded RGB response. With selection=all, every load is fitting data; no held-out claim.')
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'gain_fit.json').write_text(json.dumps(fit_report,indent=2))
        if args.candidate_depth_gains:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            scores={round(t['gain'],12):t for t in trials}
            cells=[]
            for ratio,pairs in joint_pairs.items():
                for pair in pairs:
                    cells.append(dict(**pair,relative_gain=ratio,score=scores[ratio][args.selection],
                                      cases=scores[ratio]['cases']))
            ks,ds=sorted(args.candidate_k_refs),sorted(args.candidate_depth_gains)
            matrix=np.array([[scores[round(args.source_k_ref/k*d/args.source_depth_gain,12)][args.selection]
                              for k in ks] for d in ds])
            fig,ax=plt.subplots(figsize=(10,6))
            im=ax.imshow(matrix,origin='lower',aspect='auto',cmap='viridis_r')
            ax.set_xticks(range(len(ks)),[f'{k:g}' for k in ks],rotation=45)
            ax.set_yticks(range(len(ds)),[f'{d:g}' for d in ds])
            ax.set_xlabel('k_ref (N/m)');ax.set_ylabel('depth_gain')
            ax.set_title('Six real nut loads: normalized RGB RMSE (lower is better)')
            fig.colorbar(im,ax=ax,label='Mean normalized RMSE');fig.tight_layout()
            fig.savefig(args.output/'joint_error_surface.png',dpi=160);plt.close(fig)
            (args.output/'joint_grid.json').write_text(json.dumps(dict(cells=cells,
                selected_k_ref=effective_k(args.relative_gain),selected_depth_gain=args.target_depth_gain,
                note='Saved-force/depth replay assumes no change to active force samples or stiffness caps. Same ratios share rendered predictions.'),indent=2))
    rows, crops, clean_rows, report = [], [], [], []
    for case in CASES:
        src = args.input_root/'normal_force'/case/'vitacsim'
        dst = args.output/'normal_force'/case/'vitacsim'
        old_height = np.load(src/'tactile_height_corrected.npy')
        height = old_height * args.relative_gain
        clean = renderer.render(torch.as_tensor(height, device=args.device)[None])[0]
        displacement = np.load(src/'tactile_marker_displacement.npy')
        rgb = marker.draw_markers_on_image(clean, marker.rest_xy + torch.as_tensor(displacement,device=args.device)).cpu().numpy()
        save(dst/'tactile_rgb_corrected.png',rgb)
        save(dst/'tactile_rgb_marker_free.png',clean.cpu().numpy())
        np.save(dst/'tactile_height_corrected.npy',height)
        np.save(dst/'tactile_marker_displacement.npy',displacement)
        assert np.allclose(height,old_height*args.relative_gain,rtol=1e-6,atol=0)
        real = _load_rgb(Path('data/calibration/tactile/real/normal_force')/case/'rgb.png')
        selected_cfg=renderer.cfg
        renderer.cfg=cfg
        old_clean_tensor = renderer.render(torch.as_tensor(old_height*args.baseline_gain,device=args.device)[None])[0]
        renderer.cfg=selected_cfg
        before = marker.draw_markers_on_image(old_clean_tensor,marker.rest_xy+torch.as_tensor(displacement,device=args.device)).cpu().numpy()
        labels = [case+' real',f'previous gain={args.baseline_gain:g}',f'gain={args.relative_gain:g}']
        if args.candidate_k_refs is not None:
            labels[-1]=f'k_ref={effective_k(args.relative_gain):.0f}, gain={args.target_depth_gain:g}'
        if args.baseline_k_ref is not None:
            labels[1]=f'k_ref={args.baseline_k_ref:g}, gain={args.baseline_depth_gain:g}'
        images = [real,before,rgb]
        rows.append(np.hstack([_label(im,s) for im,s in zip(images,labels)]))
        crops.append(np.hstack([_label(im[240:460,90:310],s) for im,s in zip(images,labels)]))
        old_clean = old_clean_tensor.cpu().numpy()
        clean_rows.append(np.hstack([_label(im[240:460,90:310],s) for im,s in
                         zip([old_clean,clean.cpu().numpy()],labels[1:])]))
        report.append(dict(case=case,peak_input_mm=float(old_height.max()*1000),
                           peak_before_mm=float(old_height.max()*args.baseline_gain*1000),
                           peak_after_mm=float(height.max()*1000)))
    save(args.output/'all_masses_full.png',np.vstack(rows))
    save(args.output/'all_masses_crops.png',np.vstack(crops))
    save(args.output/'all_masses_marker_free.png',np.vstack(clean_rows))
    (args.output/'summary.json').write_text(json.dumps(dict(input_root=str(args.input_root),
        relative_gain=args.relative_gain,baseline_gain=args.baseline_gain,cases=report,
        k_ref=effective_k(args.relative_gain),depth_gain=args.target_depth_gain,
        note='Saved-depth renderer replay; optics fixed; saved marker motion retained; PhysX not rerun.'),indent=2))
    print(json.dumps(report),flush=True)
    app.close()


if __name__=='__main__':
    main()
