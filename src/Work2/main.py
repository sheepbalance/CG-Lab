#執行指令 uv run -m src.Work2.main
import taichi as ti
import numpy as np

# 使用 gpu 后端
ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000  # 曲线采样点数量

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# GUI 绘制数据缓冲池
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# 用于存放曲线坐标的 GPU 缓冲区
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

def de_casteljau(points, t):
    """纯 Python 递归实现 De Casteljau 算法"""
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])

def main():
    window = ti.ui.Window("Bezier Curve - Drag to Edit", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    dragging_idx = -1  # 记录当前正在拖动哪个点的索引

    while window.running:
        # --- 1. 处理事件 (点击) ---
        events = window.get_events(ti.ui.PRESS)
        for e in events:
            if e.key == ti.ui.LMB: 
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
            elif e.key == 'c': 
                control_points = []
                dragging_idx = -1

        # --- 2. 处理拖动逻辑 (右键按下状态) ---
        if window.is_pressed(ti.ui.RMB):
            curr_mouse_pos = window.get_cursor_pos()
            
            # 如果还没开始拖动，先找最近的点
            if dragging_idx == -1 and len(control_points) > 0:
                min_dist = 1e9
                for i, pt in enumerate(control_points):
                    dist = (pt[0] - curr_mouse_pos[0])**2 + (pt[1] - curr_mouse_pos[1])**2
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                
                # 设定一个选择门槛 (距离平方 < 0.001)
                if min_dist < 0.001:
                    dragging_idx = closest_idx
            
            # 如果正在拖动，实时更新坐标
            if dragging_idx != -1:
                control_points[dragging_idx] = curr_mouse_pos
        else:
            # 放开右键时，重置拖动索引
            dragging_idx = -1

        # --- 3. 渲染逻辑 ---
        clear_pixels()
        current_count = len(control_points)
        
        if current_count >= 2:
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                curve_points_np[t_int] = de_casteljau(control_points, t)
            
            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(NUM_SEGMENTS + 1)
                    
        canvas.set_image(pixels)
        
        # 绘制辅助 UI (点和线)
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            
            # 正在拖动的点变黄色，普通点红色
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))
            
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))
        
        window.show()

if __name__ == '__main__':
    main()
