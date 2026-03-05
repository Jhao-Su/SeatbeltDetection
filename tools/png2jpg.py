import os
import sys
from PIL import Image

def convert_png_to_jpg(input_path, output_path=None, quality=95):
    """
    将 PNG 图片转换为 JPG 格式。
    
    参数:
    input_path: 输入的 PNG 文件路径
    output_path: 输出的 JPG 文件路径 (如果为 None，则自动生成同名 jpg 文件)
    quality: JPG 保存质量 (1-100)，默认 95
    """
    try:
        # 打开图片
        with Image.open(input_path) as img:
            # 获取文件名和目录
            dir_name = os.path.dirname(input_path)
            base_name = os.path.basename(input_path)
            name_without_ext = os.path.splitext(base_name)[0]
            
            # 如果未指定输出路径，则在原目录下生成同名 jpg
            if output_path is None:
                output_path = os.path.join(dir_name, f"{name_without_ext}.jpg")
            
            # 处理透明度 (PNG 有 Alpha 通道，JPG 没有)
            # 如果图片模式包含 'A' (Alpha)，则将其转换为 RGB，背景设为白色
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                # 创建一个白色背景的图像
                background = Image.new("RGB", img.size, (255, 255, 255))
                # 将原图粘贴到白色背景上，使用原图的 alpha 通道作为 mask
                background.paste(img, mask=img.split()[3] if img.mode == "RGBA" else img.split()[-1])
                img = background
            elif img.mode != "RGB":
                # 其他非 RGB 模式 (如灰度) 也强制转为 RGB 以兼容 JPG
                img = img.convert("RGB")
            
            # 保存为 JPG
            img.save(output_path, "JPEG", quality=quality)
            print(f"成功: {input_path} -> {output_path}")
            return True

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{input_path}'")
        return False
    except Exception as e:
        print(f"转换失败 '{input_path}': {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  单文件转换: python png_to_jpg.py <输入文件.png> [输出文件.jpg]")
        print("  批量转换:   python png_to_jpg.py --batch")
        print("\n示例:")
        print("  python png_to_jpg.py image.png")
        print("  python png_to_jpg.py image.png output.jpg")
        print("  python png_to_jpg.py --batch")
        return

    if sys.argv[1] == "--batch":
        # 批量处理当前目录下的所有 .png 文件
        current_dir = os.getcwd()
        files = [f for f in os.listdir(current_dir) if f.lower().endswith('.png')]
        
        if not files:
            print("当前目录下没有找到 .png 文件。")
            return

        print(f"找到 {len(files)} 个 PNG 文件，开始批量转换...")
        success_count = 0
        for filename in files:
            full_path = os.path.join(current_dir, filename)
            if convert_png_to_jpg(full_path):
                success_count += 1
        
        print(f"\n批量转换完成: {success_count}/{len(files)} 成功。")

    else:
        # 单文件处理
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_png_to_jpg(input_file, output_file)

if __name__ == "__main__":
    main()