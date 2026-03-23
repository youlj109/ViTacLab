# 将 ViTacLab 推送到 GitHub

## 1. 配置 Git 用户信息（若尚未配置）

在终端执行（把邮箱和名字换成你自己的）：

```bash
git config --global user.email "your_email@example.com"
git config --global user.name "Your Name"
```

仅在本仓库设置可省略 `--global`：

```bash
cd /home/youlinjing/Code/lightwheel/IsaacLab_510/ViTacLab
git config user.email "your_email@example.com"
git config user.name "Your Name"
```

## 2. 创建首次提交

```bash
cd /home/youlinjing/Code/lightwheel/IsaacLab_510/ViTacLab
git commit -m "Initial commit: ViTacLab with Forge tasks, visual disturbance, object randomization"
```

## 3. 在 GitHub 上创建新仓库

1. 打开 https://github.com/new
2. 填写仓库名（例如 `ViTacLab`）
3. 选择 Public/Private，**不要**勾选 “Add a README” 或 “Add .gitignore”（本地已有）
4. 点击 “Create repository”

## 4. 添加远程并推送

创建好仓库后，GitHub 会显示仓库 URL，例如 `https://github.com/你的用户名/ViTacLab.git` 或 `git@github.com:你的用户名/ViTacLab.git`。

在本地执行（**把下面的 URL 换成你的仓库地址**）：

```bash
cd /home/youlinjing/Code/lightwheel/IsaacLab_510/ViTacLab
git remote add origin https://github.com/你的用户名/ViTacLab.git
git branch -M main
git push -u origin main
```

若使用 SSH：

```bash
git remote add origin git@github.com:你的用户名/ViTacLab.git
git branch -M main
git push -u origin main
```

## 5. 可选：把 assets 加入版本库

当前 `source/ViTacLab/ViTacLab/assets/` 未纳入提交。若需要把非 USD 资源（如 README、配置）也推上去，可执行：

```bash
git add source/ViTacLab/ViTacLab/assets/
git status   # 确认新增文件（.gitignore 会排除 *.usd / *.usda）
git commit -m "Add assets (non-USD files)"
git push
```

注意：`.gitignore` 已忽略 `**/*.usd`、`**/*.usda` 等，大体积 USD 不会进仓库。
