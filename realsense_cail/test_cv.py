import cv2
import numpy as np

def test_callback(event, x, y, flags, param):
    """一个简单的回调函数，用于测试"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"鼠标成功点击，坐标: ({x}, {y})")

def run_test():
    """运行最简化OpenCV窗口测试"""
    window_name = "OpenCV 环境测试"
    print(f"1. 正在创建窗口: '{window_name}'")
    
    try:
        cv2.namedWindow(window_name)
        print("2. 窗口已声明，调用waitKey(1)强制刷新GUI...")
        cv2.waitKey(1)
        
        print("3. 正在设置鼠标回调函数...")
        cv2.setMouseCallback(window_name, test_callback)
        print("4. 回调函数设置成功。")

    except cv2.error as e:
        print(f"\n!!! 在初始化过程中发生错误 !!!")
        print(f"错误详情: {e}")
        print("这很可能是一个环境问题，请检查您的OpenCV安装和显示环境设置。")
        return

    # 创建一个黑色图像用于显示
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, 'Click here. Press Q to quit.', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("测试程序已退出。")

if __name__ == '__main__':
    run_test()