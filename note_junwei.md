# note
+ 安装
```
    (base) junweil@office-precognition:~/projects/wbc_manipulation$ git clone https://github.com/JunweiLiang/openpi
    (base) junweil@office-precognition:~/projects/wbc_manipulation/openpi$ git submodule update --init --recursive

    UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple" GIT_LFS_SKIP_SMUDGE=1 uv sync
    UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple" GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .


    # 下载模型，需要google cloud

        sudo apt-get update
        sudo apt-get install apt-transport-https ca-certificates gnupg curl
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
        sudo apt-get update && sudo apt-get install google-cloud-cli

        (base) junweil@office-precognition:~/projects/wbc_manipulation$ gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base .

```
