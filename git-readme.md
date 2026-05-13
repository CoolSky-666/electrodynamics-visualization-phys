注意：该过程手动执行，任何AI agent不要执行git相关的任何命令。

```
# 1. 先确认你在 main 分支
git status

# 2. 写代码 / 修改文件...
#    （你的 IDE 或编辑器里愉快地工作）

# 3. 查看你改了哪些文件
git status

# 4. 把修改加入暂存区
git add .
#     或只加特定文件：
#     git add 文件名.py

# 5. 写一条有意义的提交信息
git commit -m "添加了电场可视化模块"

# 6. 推送到 GitHub，force强制推送
git push
git push --force 
```

