# 远程服务器连接 GitHub 与上传代码

本文记录在远程 Linux 服务器上通过 SSH 连接 GitHub，并上传当前 Z-Image 项目的流程。

本项目示例：

```text
项目目录：/home/haoyuzhao/code/Z-Image
GitHub 用户名：gulucaptain
仓库地址：git@github.com:gulucaptain/Z-Image.git
远程名称：github
目标分支：main
```

所有终端命令都在远程服务器上执行；添加公钥、创建仓库等网页操作在浏览器中完成。

## 1. 检查或生成 SSH 密钥

先检查已有密钥：

```bash
ls -al ~/.ssh
```

如果已有 `id_ed25519` 和 `id_ed25519.pub`，可以复用，跳过生成步骤。如果没有，则执行：

```bash
ssh-keygen -t ed25519 -C "你的GitHub邮箱"
```

按提示选择保存位置并设置密钥口令。默认路径为 `~/.ssh/id_ed25519`；如果提示覆盖已有密钥，不要覆盖，可以复用已有密钥或选择其他文件名。

启动 SSH agent 并加载私钥：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

如果密钥设置了口令，`ssh-add` 会要求输入该口令。新登录服务器后，可能需要再次执行这两条命令。使用自定义密钥文件名时，将命令中的路径替换为实际路径。

参考：[GitHub 官方密钥生成说明](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)。

## 2. 将公钥添加到 GitHub

查看公钥：

```bash
cat ~/.ssh/id_ed25519.pub
```

复制完整输出，然后在 GitHub 网页打开：

**头像 → Settings → SSH and GPG keys → New SSH key**

- Title：填写方便识别的服务器名称。
- Key type：选择 `Authentication Key`。
- Key：粘贴公钥内容，然后保存。

添加的是 `.pub` 公钥；`id_ed25519` 私钥保留在服务器，不上传或分享。

参考：[GitHub 官方公钥添加说明](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)。

## 3. 测试 SSH 连接

```bash
ssh -T git@github.com
```

首次连接可能询问是否信任主机。核对显示的指纹与 [GitHub 官方 SSH 指纹](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)一致后，输入 `yes`。

成功时会看到：

```text
Hi gulucaptain! You've successfully authenticated, but GitHub does not provide shell access.
```

这表示已经成功认证到 `gulucaptain` 账号。`does not provide shell access` 是正常提示：GitHub 提供 Git 操作，不提供交互式终端。即使认证成功，该测试命令也可能返回退出码 1。

参考：[GitHub 官方连接测试说明](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection)。

## 4. 创建仓库并配置项目远程地址

在 GitHub 创建 `gulucaptain/Z-Image`。首次上传已有项目时，可以创建空仓库，不勾选自动生成 README、License 或 `.gitignore`。如果仓库已有提交，按后面的合并步骤处理。

进入项目并设置提交作者：

```bash
cd /home/haoyuzhao/code/Z-Image
git config user.name "gulucaptain"
git config user.email "你的GitHub邮箱"
```

这里的姓名、邮箱是 Git 提交信息；SSH 密钥负责登录认证。

检查现有远程：

```bash
git remote -v
```

如果还没有名为 `github` 的远程，添加它：

```bash
git remote add github git@github.com:gulucaptain/Z-Image.git
```

如果 `github` 已经存在，更新地址：

```bash
git remote set-url github git@github.com:gulucaptain/Z-Image.git
```

再次运行 `git remote -v` 确认。使用新的 `github` 名称可以保留项目已有的 `origin`；以下命令统一使用 `github`。如果实际使用 `origin`，替换对应名称即可。

## 5. 提交并首次上传

项目已有 `.gitignore`，会忽略模型权重、`outputs/`、`outputs2/`、`tmp/`、虚拟环境及本地敏感配置。忽略规则不会自动移除已经被 Git 跟踪的文件。

先检查并提交本地代码：

```bash
git status
git add .
git diff --cached --stat
git commit -m "Add Gradio workspace and inference experiments"
```

检查暂存内容符合预期后再提交。如果提示 `nothing to commit`，表示当前没有新的改动需要提交。

将当前提交上传到远程 `main`：

```bash
git push -u github HEAD:main
```

`HEAD:main` 明确表示将当前本地分支推送到远程 `main`，即使本地分支名称不同也适用。

## 6. 为什么还要求输入用户名和密码？

如果提示如下：

```text
Username for 'https://github.com':
```

说明当前 Git 操作使用的是 HTTPS 地址。SSH 认证成功不会自动修改仓库的远程地址。

检查并改为 SSH 地址，再显式推送到该远程：

```bash
git remote -v
git remote set-url github git@github.com:gulucaptain/Z-Image.git
git push -u github HEAD:main
```

如果提示如下：

```text
Enter passphrase for key ...
```

要求输入的是 SSH 密钥口令，不是 GitHub 账号密码。通过以下命令加载密钥后，SSH agent 可以在当前会话中记住它：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

## 7. 处理 non-fast-forward 推送失败

典型报错：

```text
! [rejected] HEAD -> main (non-fast-forward)
error: failed to push some refs to 'github.com:gulucaptain/Z-Image.git'
```

这说明远程 `main` 中有当前本地历史尚未包含的提交。例如，在 GitHub 创建仓库时自动添加了 README，或者其他地方已经推送过内容。需要先合并远程历史。[GitHub 官方说明](https://docs.github.com/en/get-started/using-git/dealing-with-non-fast-forward-errors)。

先执行 `git status`，确保要保留的本地修改已提交，再获取远程提交并合并：

```bash
git fetch github
git merge github/main --allow-unrelated-histories --no-edit
```

`--allow-unrelated-histories` 用于本地项目与远程仓库各自初始化、没有共同祖先的情况；`--no-edit` 使用默认合并提交说明。[Git merge 官方文档](https://git-scm.com/docs/git-merge)。

**只有合并成功后**，才执行：

```bash
git push -u github HEAD:main
```

### 如果出现合并冲突

出现 `CONFLICT` 时，先查看冲突文件：

```bash
git status
```

打开冲突文件，处理如下标记中的两边内容，保留需要的最终版本并删除标记：

```text
<<<<<<< HEAD
本地内容
=======
远程内容
>>>>>>> github/main
```

将实际解决的文件加入暂存区，例如冲突文件是 README：

```bash
git add README.md
git commit -m "Merge GitHub main and resolve conflicts"
git push -u github HEAD:main
```

有多个冲突文件时，需要全部解决并暂存，再执行提交。不要在未解决冲突时继续推送。如果提示本地修改会被覆盖，应先提交这些修改，再重试合并。

## 8. 后续更新代码

每次完成修改后：

```bash
cd /home/haoyuzhao/code/Z-Image
git status
git add .
git diff --cached --stat
git commit -m "描述这次修改"
git push github HEAD:main
```

如果远程又有新提交导致推送被拒绝，重复第 7 节的获取、合并和推送流程。
