import os
from datasets import load_dataset
from tqdm import tqdm

def download_and_prepare_dataset():
    # 数据集在 Hugging Face Hub 上的 ID
    repo_id = "peteromallet/InScene-Dataset"
    
    # 本地保存路径
    base_path = os.path.join("datasets", "inscene_dataset")
    control_path = os.path.join(base_path, "control")
    target_path = os.path.join(base_path, "target")
    
    # 创建目录
    os.makedirs(control_path, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    
    print(f"正在从 '{repo_id}' 下载数据集...")
    # 加载数据集，如果网络有问题可能需要一些时间
    # trust_remote_code=True 是某些数据集所必需的
    dataset = load_dataset(repo_id, trust_remote_code=True)
    
    # 我们只处理训练集 'train'
    train_split = dataset['train']
    
    print("数据集下载完成，开始处理图像和提示词...")
    # 使用 tqdm 显示进度条
    for i, item in enumerate(tqdm(train_split)):
        try:
            # 获取图像和提示词
            control_image = item['control_image']
            target_image = item['target_image']
            prompt = item['prompt']
            
            # 使用索引作为文件名，确保唯一性
            base_filename = f"{i:06d}"
            
            # 保存图像
            control_image.save(os.path.join(control_path, f"{base_filename}.jpg"))
            target_image.save(os.path.join(target_path, f"{base_filename}.jpg"))
            
            # 保存提示词为同名 .txt 文件
            with open(os.path.join(target_path, f"{base_filename}.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)

        except Exception as e:
            print(f"处理第 {i} 个项目时出错: {e}")
            
    print("="*50)
    print("数据集处理完成!")
    print(f"控制图像保存在: {control_path}")
    print(f"目标图像和提示词保存在: {target_path}")
    print("="*50)

if __name__ == "__main__":
    download_and_prepare_dataset()
