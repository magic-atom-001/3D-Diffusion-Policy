import os
import hydra
import torch
from omegaconf import OmegaConf
from train import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from tqdm import tqdm
import numpy as np

@hydra.main(
    version_base=None,
    config_path="diffusion_policy_3d/config"
)
def main(cfg):
    print("🔧 初始化 Workspace...")
    workspace = TrainDP3Workspace(cfg)
    workspace.load_checkpoint()

    print("📂 加载数据集...")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = workspace.ema_model if cfg.training.use_ema else workspace.model
    model.eval()
    model.cuda()

    pred_actions = []
    gt_actions = []

    print("🤖 开始逐帧推理...")
    for i, sample in enumerate(tqdm(dataloader, desc="推理中")):
        sample = dict_apply(sample, lambda x: x.cuda())

        obs = {
            "point_cloud": sample["obs"]["point_cloud"],
            "agent_pos": sample["obs"]["agent_pos"]
        }

        with torch.no_grad():
            out = model.predict_action(obs)
            pred_action = out["action_pred"].cpu().numpy()
            pred_actions.append(pred_action)

            if "action" in sample:
                gt_action = sample["action"].cpu().numpy()
                gt_actions.append(gt_action)

                # ✅ 打印前几个样本的对比信息（可调整数量）
                if i < 200:
                    print(f"\n--- 第 {i+1} 帧 ---")
                    print(f"预测动作 (pred): {np.round(pred_action, 3)}")
                    print(f"真实动作 (gt):   {np.round(gt_action, 3)}")

    pred_actions = np.concatenate(pred_actions, axis=0)
    gt_array = None
    if len(gt_actions) > 0:
        gt_array = np.concatenate(gt_actions, axis=0)

    save_path = os.path.join(workspace.output_dir, "inference_result.npz")
    print(f"💾 保存预测结果到：{save_path}")
    np.savez_compressed(save_path, pred=pred_actions, gt=gt_array)

    print("✅ 推理完成。")


if __name__ == "__main__":
    main()
