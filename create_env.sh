Root=/mnt/coai_nas/qianhu/github/Codev-Bench/
SourceCodeRoot=$Root/Source_Code

# Setup the execution environment for contrastors
for RepoName in UHGEval UniRef camp_zipnerf microagents microsearch nlm-ingestor ollama-python open-iris searcharray tanuki_py
do
    cd $SourceCodeRoot/$RepoName
    python3.10 -m venv myenv && source myenv/bin/activate
    pip install pytest pytest-runner pytest-timeout -i https://pypi.tuna.tsinghua.edu.cn/simple/
    pip install -r true_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
    deactivate
done