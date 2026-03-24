import cv2
import os
import shutil
import numpy as np
from PIL import Image
import requests
import json
import time
import re
import sys

from cozepy import COZE_CN_BASE_URL
from cozepy import Coze, TokenAuth, Message, ChatEventType, MessageContentType

# ===================== 核心配置（重点：多智能体配置区，后续加新智能体只需改这里） =====================
COZE_AUTH_TOKEN = ""# 身份认证令牌
USER_ID = ""  # 用户id
INPUT_FOLDER = "input_images"  # 原始图片文件夹
PREPROCESS_FOLDER = "preprocessed_images"  # 预处理图片文件夹
OUTPUT_FOLDER = "processed_images"  # 处理后效果图文件夹
RECYCLE_FOLDER = "processed_recycle"  # 统一回收站文件夹
MAX_WAIT_TIME = 60  # 智能体回复超时时间
BASE_SIZE = 512  # 尺寸比例基准边长

# 多智能体配置字典（后续添加新智能体，只需在这里新增条目）
# 格式：{智能体标识: {"bot_id": "智能体ID", "keywords": [匹配关键词列表], "desc": "智能体描述"}}
AGENT_CONFIG = {
    # 已有的图像处理智能体（示例）
    "image_process": {
        "bot_id": "7544332598131589130",
        "keywords": ["图片", "图像", "美颜", "去水印", "画质提升", "尺寸", "头像", "壁纸", "修图"],
        "desc": "图像处理智能体（美颜/去水印/画质提升/尺寸调整）"
    },
    # 预留视频处理智能体（你后续只需替换bot_id和关键词即可）
    "video_process": {
        "bot_id": "",  # 替换为你的视频处理智能体ID
        "keywords": ["视频", "剪辑", "调色", "去水印", "帧率", "分辨率", "转码"],
        "desc": "视频处理智能体（剪辑/调色/去水印/分辨率调整）"
    },
    # 可继续添加更多智能体，比如：
    # "text_process": {
    #     "bot_id": "",
    #     "keywords": ["文本", "翻译", "总结", "改写"],
    #     "desc": "文本处理智能体（翻译/总结/改写）"
    # }
}
# ======================================================================================================

# 初始化所有文件夹（自动创建缺失文件夹）
for folder in [INPUT_FOLDER, PREPROCESS_FOLDER, OUTPUT_FOLDER, RECYCLE_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"✅ 已创建文件夹：{folder}")

# 初始化扣子SDK（通用，无需按智能体区分）
coze = Coze(auth=TokenAuth(token=COZE_AUTH_TOKEN), base_url=COZE_CN_BASE_URL)

# 预设尺寸比例模板（含用途说明）
PRESET_RATIOS = {
    "1": ("1:1 正方形，头像", 1, 1),
    "2": ("2:3 社交媒体，自拍", 2, 3),
    "3": ("3:4 经典比例，拍照", 3, 4),
    "4": ("4:3 文章配图，插画", 4, 3),
    "5": ("9:16 手机壁纸，人像", 9, 16),
    "6": ("16:9 桌面壁纸，风景", 16, 9),
    "7": ("自定义尺寸", 0, 0)
}


def check_input_images():
    """检测input_images是否有可处理的图片"""
    image_ext = ('.jpg', '.jpeg', '.png')
    has_image = any(
        f.lower().endswith(image_ext) for f in os.listdir(INPUT_FOLDER)
    )
    return has_image


def wait_user_choice():
    """等待用户选择：输入exit退出，输入其他重新运行（适配PyCharm）"""
    print("\n=====================================")
    print("请输入操作：输入【exit】退出程序 | 输入任意内容重新运行")
    print("=====================================")
    choice = input("你的选择：").strip().lower()
    if choice == 'exit':
        print("\n👋 程序已退出")
        sys.exit(0)
    else:
        print("\n🔄 开始重新运行程序...")
        return True


def choose_image_ratio():
    """让用户选择预处理的图片尺寸比例，返回(目标尺寸, 比例描述)"""
    print("\n===== 选择预处理图片尺寸比例 =====")
    for key, (ratio_desc, _, _) in PRESET_RATIOS.items():
        print(f"{key}. {ratio_desc}")

    while True:
        choice = input("\n请输入比例序号（1-7）：").strip()
        if choice in PRESET_RATIOS:
            ratio_desc, w_ratio, h_ratio = PRESET_RATIOS[choice]

            # 自定义尺寸逻辑
            if choice == "7":
                while True:
                    try:
                        width = int(input("请输入自定义宽度（像素）：").strip())
                        height = int(input("请输入自定义高度（像素）：").strip())
                        if width > 0 and height > 0:
                            target_size = (width, height)
                            ratio_desc = f"{width}:{height} 自定义尺寸"
                            print(f"✅ 选择自定义尺寸：{width}×{height}")
                            return target_size, ratio_desc
                        else:
                            print("⚠️  宽度/高度必须大于0")
                    except ValueError:
                        print("⚠️  请输入有效的数字")

            # 预设比例计算尺寸
            else:
                if w_ratio >= h_ratio:
                    width = BASE_SIZE
                    height = int(BASE_SIZE * h_ratio / w_ratio)
                else:
                    height = BASE_SIZE
                    width = int(BASE_SIZE * w_ratio / h_ratio)
                target_size = (width, height)
                print(f"✅ 选择{ratio_desc}，对应尺寸：{width}×{height}")
                return target_size, ratio_desc
        else:
            print("⚠️  输入无效，请输入1-7的序号")


def confirm_preprocess():
    """让用户选择是否预处理，返回(是否预处理, 目标尺寸, 比例描述)"""
    while True:
        choice = input("\n是否需要对原始图片进行OpenCV预处理？(y/n，默认y)：").strip().lower()
        if not choice or choice == 'y':
            target_size, ratio_desc = choose_image_ratio()
            print("✅ 选择：进行预处理")
            return True, target_size, ratio_desc
        elif choice == 'n':
            print("✅ 选择：不进行预处理，直接使用原始图片")
            return False, (0, 0), ""
        else:
            print("⚠️  输入无效，请输入y/n")


def batch_preprocess_images(input_folder, output_folder, target_size, ratio_desc):
    """预处理图片，返回(预处理路径列表, 原始-预处理路径映射)"""
    preprocessed_paths = []
    raw_to_pre_map = {}

    for img_name in os.listdir(input_folder):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            raw_img_path = os.path.join(input_folder, img_name)
            try:
                img = cv2.imread(raw_img_path)
                if img is None:
                    print(f"❌ 跳过无效图片：{img_name}")
                    continue
                # 按选择的尺寸缩放
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                # BGR转RGB（适配扣子上传）
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                pre_img_name = f"pre_{img_name}"
                pre_img_path = os.path.join(output_folder, pre_img_name)
                Image.fromarray(img_rgb).save(pre_img_path)

                preprocessed_paths.append(pre_img_path)
                raw_to_pre_map[raw_img_path] = pre_img_path
                # 提取纯比例信息（如"1:1"）
                ratio = ratio_desc.split(' ')[0]
                print(f"✅ 预处理完成：{img_name} → {pre_img_name}（{ratio}，{target_size[0]}×{target_size[1]}）")
            except Exception as e:
                print(f"❌ 处理{img_name}失败：{str(e)}")
                continue

    print(f"\n📊 预处理完成！共处理{len(preprocessed_paths)}张有效图片")
    return preprocessed_paths, raw_to_pre_map


def upload_file_to_coze(file_path):
    """上传文件到扣子，返回file_id/None"""
    url = "https://api.coze.cn/v1/files/upload"
    headers = {"Authorization": f"Bearer {COZE_AUTH_TOKEN}"}
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        try:
            response = requests.post(url, headers=headers, files=files, timeout=30)
            response.raise_for_status()
            result = response.json()
            if result.get("code") == 0 and "data" in result:
                return result["data"]["id"]
            else:
                print(f"❌ 上传{os.path.basename(file_path)}失败：{result.get('msg')}")
                return None
        except Exception as e:
            print(f"❌ 上传{os.path.basename(file_path)}异常：{str(e)}")
            return None


def download_coze_image(image_url, save_path):
    """带鉴权下载扣子/字节图片，返回是否成功"""
    headers = {"Authorization": f"Bearer {COZE_AUTH_TOKEN}"}
    try:
        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✅ 效果图下载成功：{save_path}")
        return True
    except Exception as e:
        print(f"❌ 下载{image_url}失败：{str(e)}")
        return False


def move_file_to_recycle(file_path, recycle_folder):
    """移动文件到回收站，自动处理重名"""
    if not os.path.exists(file_path):
        print(f"⚠️  文件不存在，跳过移动：{file_path}")
        return

    file_name = os.path.basename(file_path)
    target_path = os.path.join(recycle_folder, file_name)

    # 重名处理：添加时间戳后缀
    if os.path.exists(target_path):
        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(file_name)
        new_file_name = f"{name}_{timestamp}{ext}"
        target_path = os.path.join(recycle_folder, new_file_name)

    try:
        shutil.move(file_path, target_path)
        print(f"✅ 归档到回收站：{file_name} → {os.path.basename(target_path)}")
    except Exception as e:
        print(f"❌ 移动{file_name}失败：{str(e)}")


def match_agent_by_instruction(instruction):
    """根据用户指令关键词匹配对应的智能体，返回(bot_id, 智能体描述)"""
    instruction_lower = instruction.lower()
    matched_agents = []

    # 遍历所有智能体，匹配关键词
    for agent_key, agent_info in AGENT_CONFIG.items():
        # 跳过未配置bot_id的智能体（如预留的视频处理）
        if not agent_info["bot_id"]:
            continue
        # 检查指令是否包含该智能体的任意关键词
        for keyword in agent_info["keywords"]:
            if keyword in instruction_lower:
                matched_agents.append((agent_info["bot_id"], agent_info["desc"]))
                break  # 匹配到一个关键词即可，避免重复

    # 处理匹配结果
    if len(matched_agents) == 0:
        # 无匹配智能体，提示用户并退出
        print("\n❌ 未匹配到可用的智能体！")
        print("当前支持的智能体及关键词：")
        for agent_key, agent_info in AGENT_CONFIG.items():
            if agent_info["bot_id"]:
                print(f"- {agent_info['desc']}：关键词={','.join(agent_info['keywords'])}")
        sys.exit(1)
    elif len(matched_agents) >= 1:
        # 匹配到多个/一个，默认选第一个（可扩展为让用户选择）
        bot_id, agent_desc = matched_agents[0]
        print(f"\n✅ 匹配到智能体：{agent_desc}")
        return bot_id, agent_desc


def process_images():
    """核心图片处理逻辑"""
    # 步骤1：检测input_images是否有图片
    while not check_input_images():
        print("\n⚠️  input_images文件夹中无可用图片！")
        print("请将图片放入input_images文件夹后重试")
        print("\n输入【exit】退出 | 输入任意内容重新检测")
        choice = input("你的选择：").strip().lower()
        if choice == 'exit':
            print("\n👋 程序已退出")
            sys.exit(0)

    # 步骤2：用户选择是否预处理 + 选择尺寸
    need_preprocess, target_size, ratio_desc = confirm_preprocess()
    raw_to_pre_map = {}

    if need_preprocess:
        # 预处理图片
        image_paths, raw_to_pre_map = batch_preprocess_images(
            INPUT_FOLDER, PREPROCESS_FOLDER, target_size, ratio_desc
        )
    else:
        # 不预处理，直接使用原始图片
        image_paths = [
            os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        raw_to_pre_map = {img_path: img_path for img_path in image_paths}
        print(f"\n📊 共读取{len(image_paths)}张原始图片")

    if not image_paths:
        print("❌ 无有效图片可处理")
        return

    # 步骤3：批量上传图片
    file_id_list = []
    print("\n===== 开始批量上传图片 =====")
    for img_path in image_paths:
        file_id = upload_file_to_coze(img_path)
        if file_id:
            file_id_list.append(file_id)
            print(f"✅ 已上传：{os.path.basename(img_path)}，file_id：{file_id[:10]}...")
        time.sleep(2)  # 避免上传频率过高

    if not file_id_list:
        print("❌ 无图片上传成功")
        return

    # 步骤4：输入处理指令 + 自动匹配智能体
    user_request = input("\n请输入你的处理指令：")
    # 核心：根据指令匹配智能体
    bot_id, agent_desc = match_agent_by_instruction(user_request)

    # 构造多模态消息
    multi_msg = [{"type": "text", "text": user_request}]
    for file_id in file_id_list:
        multi_msg.append({"type": "image", "file_id": file_id})
    multi_msg_json = json.dumps(multi_msg)

    print(f"\n===== 发送{len(file_id_list)}张图+指令给【{agent_desc}】 =====")
    print(f"你：{user_request}（附带{len(file_id_list)}张图片）")
    print("智能体回复中...（最多等待60秒）")

    full_reply = ""
    try:
        start_time = time.time()
        for event in coze.chat.stream(
                bot_id=bot_id,  # 动态使用匹配到的智能体ID
                user_id=USER_ID,
                additional_messages=[
                    Message(
                        role="user",
                        content_type=MessageContentType.OBJECT_STRING,
                        content=multi_msg_json,
                        type="question"
                    )
                ]
        ):
            if time.time() - start_time > MAX_WAIT_TIME:
                print(f"\n❌ 等待超时（超过{MAX_WAIT_TIME}秒）")
                break   
            if event.event == ChatEventType.CONVERSATION_MESSAGE_DELTA:
                reply_chunk = event.message.content
                full_reply += reply_chunk
                print(reply_chunk, end="", flush=True)
            if event.event == ChatEventType.CONVERSATION_CHAT_COMPLETED:
                print(f"\n✅ 对话完成（耗时：{int(time.time() - start_time)}秒）")
                break
    except Exception as e:
        print(f"\n❌ 发消息失败：{str(e)}")
        return

    # 步骤5：提取并下载所有图片链接（兼容任意域名）
    image_urls = re.findall(r'https?://[^\s\)]+', full_reply)
    if image_urls:
        print(f"\n===== 下载{len(image_urls)}张效果图 =====")
        for idx, url in enumerate(image_urls, 1):
            save_name = f"processed_{idx}.jpg"
            save_path = os.path.join(OUTPUT_FOLDER, save_name)
            download_coze_image(url, save_path)
    else:
        print("⚠️  未提取到图片链接")

    # 步骤6：归档已处理文件到回收站
    print(f"\n===== 归档已处理文件 =====")
    for raw_path, pre_path in raw_to_pre_map.items():
        move_file_to_recycle(raw_path, RECYCLE_FOLDER)
        if need_preprocess and raw_path != pre_path:
            move_file_to_recycle(pre_path, RECYCLE_FOLDER)


# 主程序：循环运行
if __name__ == "__main__":
    print("🚀 多智能体图片/视频处理程序已启动")
    while True:
        process_images()
        wait_user_choice()
