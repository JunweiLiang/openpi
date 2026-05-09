# note
+ 安装
```
    (base) junweil@office-precognition:~/projects/wbc_manipulation$ git clone https://github.com/JunweiLiang/openpi
    (base) junweil@office-precognition:~/projects/wbc_manipulation/openpi$ git submodule update --init --recursive

    UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple" GIT_LFS_SKIP_SMUDGE=1 uv sync
    UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple" GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

```
+ finetune前准备
```
    1. 数据集, 需要lerobot v2
        # 用 humanoid_teleop/g1_realrobot/convert_unitree_json_to_lerobot_pi.py
        # 和gr00t用的相比，图像维度顺序不太一样
        # 诗蕙的 convert_unitree_json_to_lerobot_shihui.py 针对unloading会拿qpos[4]作为gripper
        # openpi的安装，lerobot 版本是0.1.0, python3.11
        # tv环境，也就是以下这些，是0.4.1, python3.10

        # 先pick_up_object_from_ground

            $ rm -rf ~/.cache/huggingface/datasets/
            $ rm -rf ~/.cache/huggingface/lerobot/junweiliang/piwbc_pick_up_object_from_ground*
            $ rm -rf ~/.cache/huggingface/lerobot/junweiliang/piwbc*


                # pi 需要再lerobot的state数据维度就确定好，而且需要relative/absolute分好前后

                # 30 fps, arms/waist/trigger/loco_cmd 23-dim

            (tv) junweil@office-precognition:~/projects/huawei_data$ python ~/projects/humanoid_teleop/g1_realrobot/convert_unitree_json_to_lerobot_pi.py --raw-dir wbc_task5_lerobotv2/ --repo-id junweiliang/piwbc_pick_up_object_from_ground --downsample-factor 2 --use-future-state-as-action --valp 0.1 --repo-id-val junweiliang/piwbc_pick_up_object_from_ground_val0.1 --tasks pick_up_object_from_ground

            # v3转v2.1; (还需要额外回退：List转成Sequence)

                (tv) junweil@office-precognition:~/projects/huawei_data$ python ~/projects/humanoid_teleop/g1_realrobot/convert_v3_to_v2_openpi.py --repo-id junweiliang/piwbc_pick_up_object_from_ground

            # 现在就可以了

                junweil@office-precognition:~/projects/wbc_manipulation/openpi$ uv run scripts/cal_lerobot.py junweiliang/piwbc_pick_up_object_from_ground
                    Total samples (frames): 22096
                    Number of episodes: 55
                    Average frames per episode: 401.75

            # 5 tasks

                (tv) junweil@office-precognition:~/projects/huawei_data$ python ~/projects/humanoid_teleop/g1_realrobot/convert_unitree_json_to_lerobot_pi.py --raw-dir wbc_task5_lerobotv2/ --repo-id junweiliang/piwbc_5tasks --downsample-factor 2 --use-future-state-as-action --valp 0.1 --repo-id-val junweiliang/piwbc_5tasks_val0.1

                (tv) junweil@office-precognition:~/projects/huawei_data$ python ~/projects/humanoid_teleop/g1_realrobot/convert_v3_to_v2_openpi.py --repo-id junweiliang/piwbc_5tasks

                junweil@office-precognition:~/projects/wbc_manipulation/openpi$ uv run scripts/cal_lerobot.py junweiliang/piwbc_5tasks
                        Total samples (frames): 124560
                        Number of episodes: 252
                        Average frames per episode: 494.29

                        # 这个统计和我之前gr00t的是一样的
    2. 准备config

        2.0 数据note
            # 原生teleop 采集的json:
                # RGB的名字是"color_0"，lerobot转换成"observation.images.cam_high"
                    #image shape是 (3, 480, 640)

                # State and Action slices are now perfectly 1:1
                # New 23D vector: Arms(14) + Waist(3) + Triggers(2) + Loco(4)


        2.1 修改添加 src/openpi/policies/g1_v1_policy.py
        2.2 修改 training/config.py
        2.3 注意 openpi默认"actions"，但是我们的数据是"action"


        # 下载模型，需要google cloud

        sudo apt-get update
        sudo apt-get install apt-transport-https ca-certificates gnupg curl
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
        sudo apt-get update && sudo apt-get install google-cloud-cli

        (base) junweil@office-precognition:~/projects/wbc_manipulation$ gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base .

        # weight path: /home/junweil/projects/wbc_manipulation/pi05_base/params


    # 训练！

```
