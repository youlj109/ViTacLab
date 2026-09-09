# ViTacLab project website

The project website is a dependency-free static site modeled after the information flow of an academic project page: paper identity, visual overview, research components, benchmark, results, resource status, and quick start. Its implementation and artwork are original to ViTacLab.

## Local preview

From the repository root, run:

```bash
python3 -m http.server 8000 --directory docs
```

Then open <http://localhost:8000/>. Relative asset paths are used so the same files work locally and at the project URL:

<https://youlj109.github.io/ViTacLab/>

## Publishing

The workflow in `.github/workflows/pages.yml` uploads `docs/` to GitHub Pages after changes land on `main`. The repository's Pages source must be set to **GitHub Actions** once; subsequent deployments are automatic.

## Maintenance map

- `index.html` — page content, metadata, and semantic structure
- `assets/site.css` — responsive layout and visual system
- `assets/site.js` — mobile navigation, reveal behavior, section state, and copy action
- `media/paper/` — figures exported from the manuscript
- `favicon.svg`, `site.webmanifest`, `robots.txt`, `sitemap.xml` — browser and crawler metadata

When a public preprint becomes available, replace both “Coming soon” paper states in `index.html`, add the official BibTeX, and update the structured metadata without guessing an arXiv identifier, DOI, year, or affiliation. Update `sitemap.xml` only when the public URL structure changes.

`index.html` uses a strict content-security policy with SHA-256 hashes for its two inline scripts. After editing the early `js` class bootstrap or the JSON-LD block, recompute the corresponding hash and replace it in the `Content-Security-Policy` meta tag. For the JSON-LD block:

```bash
perl -0777 -ne 'print $1 if /<script type="application\/ld\+json">(.*?)<\/script>/s' docs/index.html \
  | openssl dgst -sha256 -binary \
  | openssl base64 -A
```

## Design and content constraints

- Use manuscript-owned figures; do not copy RoboTwin's code or visual assets.
- Keep unavailable datasets, assets, and checkpoints visibly labeled as partial or coming soon.
- Preserve useful image alternatives, keyboard focus, horizontal figure scrolling on mobile, and reduced-motion behavior.
- Optimize any future animation as WebM/MP4 with a poster instead of adding a large GIF.
