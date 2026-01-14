# 上传到GitHub快速指南

本文档将帮助您将项目上传到GitHub。

## ✅ 已完成的准备工作

所有上传前的准备工作已完成：

1. ✅ 创建了 `.gitignore` 文件 - 自动排除大文件
2. ✅ 创建了 `requirements.txt` 文件 - 列出项目依赖
3. ✅ 修复了所有硬编码路径 - 改为相对路径
4. ✅ 更新了下载文档 - 添加百度网盘链接
5. ✅ 验证了代码语法 - 所有Python文件通过检查

## 📦 大文件已处理

以下大文件已被 `.gitignore` 排除，不会上传到GitHub：

- `trained_models/**/*.pth` (6个模型文件，每个1.2GB)
- `logs/*.log` (日志文件，共229MB)
- `datasets/` (数据集目录)
- `__pycache__/` (Python缓存)

这些文件已经在百度网盘中: https://pan.baidu.com/s/5CXLX9bODEHBSVfKVRLsmdg

## 🚀 上传步骤

### 第1步：初始化Git仓库

在项目根目录（`remote_sensing_segmentation_project/`）打开命令行：

```bash
# 初始化Git仓库
git init

# 添加所有文件（.gitignore会自动过滤大文件）
git add .

# 查看将要提交的文件（确认没有大文件）
git status
```

**重要**：检查 `git status` 输出，确保没有看到：
- ❌ `trained_models/**/model.pth`
- ❌ `logs/*.log`
- ❌ `datasets/` 目录

如果看到这些文件，说明 `.gitignore` 没有生效。

### 第2步：创建第一次提交

```bash
# 提交所有文件
git commit -m "Initial commit: DINOv3 remote sensing segmentation project

- Support for 6 remote sensing datasets (LoveDA, iSAID, Vaihingen, Potsdam, LandCover.ai, OpenEarthMap)
- DINOv3-based segmentation model
- Training and inference scripts
- Complete documentation in English and Chinese
- Trained models available on Baidu Cloud"
```

### 第3步：在GitHub创建仓库

1. 访问 https://github.com/new
2. 填写仓库信息：
   - **Repository name**: `dinov3-remote-sensing-segmentation` (或其他名字)
   - **Description**: `DINOv3-based semantic segmentation for remote sensing imagery`
   - **Public** 或 **Private** (根据需要选择)
   - ⚠️ **不要**勾选 "Add a README file"（我们已经有了）
   - ⚠️ **不要**勾选 "Add .gitignore"（我们已经有了）
3. 点击 "Create repository"

### 第4步：连接远程仓库并推送

复制GitHub显示的命令，或使用以下命令（替换为你的仓库URL）：

```bash
# 添加远程仓库
git remote add origin https://github.com/你的用户名/仓库名.git

# 推送到GitHub（首次推送使用 -u 参数）
git push -u origin main

# 如果出现分支名不是main而是master的情况
git branch -M main
git push -u origin main
```

### 第5步：验证上传

1. 在浏览器中访问你的GitHub仓库
2. 检查文件列表，应该看到：
   - ✅ README.md 和 README_CN.md
   - ✅ requirements.txt 和 .gitignore
   - ✅ `datasets/`, `models/`, `scripts/`, `tests/`, `docs/` 目录
   - ✅ 但 **不应该** 看到 `trained_models/**/model.pth` 文件
3. 点击 "commits" 查看提交历史

## 📝 后续维护

### 添加新文件或修改代码

```bash
# 查看修改的文件
git status

# 添加修改的文件
git add <文件名>
# 或添加所有修改
git add .

# 提交
git commit -m "描述你的修改"

# 推送到GitHub
git push
```

### 常用Git命令

```bash
# 查看状态
git status

# 查看修改内容
git diff

# 查看提交历史
git log --oneline

# 撤销未提交的修改
git checkout -- <文件名>

# 查看远程仓库
git remote -v
```

## ⚠️ 注意事项

### 如果不小心提交了大文件

如果你不小心提交了大文件（如 .pth 模型），需要从历史记录中移除：

```bash
# 从Git历史中移除大文件
git filter-branch --tree-filter 'rm -rf trained_models' HEAD

# 强制推送（谨慎使用！）
git push origin --force --all
```

**更简单的方法**：如果刚提交还没push，可以回退：

```bash
# 撤销最后一次提交，但保留文件修改
git reset HEAD~1

# 修改 .gitignore，确保大文件被排除

# 重新提交
git add .
git commit -m "Initial commit (fixed)"
```

### GitHub文件大小限制

- 单个文件最大: 100 MB
- 推荐单个文件大小: < 50 MB
- 仓库总大小推荐: < 1 GB

我们的项目（排除大文件后）约 < 1 MB，完全符合要求。

## 🔧 故障排查

### 问题1：push被拒绝（文件太大）

**症状**:
```
remote: error: File trained_models/xxx/model.pth is 1.20 GB; this exceeds GitHub's file size limit of 100.00 MB
```

**解决方法**:
1. 确保 `.gitignore` 正确配置
2. 从提交中移除大文件（见上面"如果不小心提交了大文件"）
3. 重新提交和推送

### 问题2：.gitignore 不生效

**症状**: `git status` 仍然显示应该被忽略的文件

**解决方法**:
```bash
# 清除Git缓存
git rm -r --cached .

# 重新添加所有文件（这次会应用.gitignore）
git add .

# 提交
git commit -m "Fix .gitignore"
```

### 问题3：推送速度很慢

**解决方法**:
- 使用代理
- 使用 GitHub Desktop 客户端
- 使用 SSH 而不是 HTTPS

```bash
# 切换到SSH（需要先配置SSH密钥）
git remote set-url origin git@github.com:用户名/仓库名.git
```

## 📚 相关文档

- [README.md](../README.md) - 项目主文档（英文）
- [README_CN.md](../README_CN.md) - 项目主文档（中文）
- [LARGE_FILES_CN.md](LARGE_FILES_CN.md) - 大文件下载说明（中文）
- [LARGE_FILES.md](LARGE_FILES.md) - 大文件下载说明（英文）

## ✅ 检查清单

上传前请确认：

- [ ] `.gitignore` 文件存在并配置正确
- [ ] `git status` 不显示大文件（*.pth, *.log等）
- [ ] README.md 中的下载链接已更新
- [ ] requirements.txt 包含所有依赖
- [ ] 代码中没有硬编码的绝对路径
- [ ] 百度网盘链接可以正常访问

---

如果遇到其他问题，请参考 [GitHub官方文档](https://docs.github.com/)。
