# src/Work4/main.py
import taichi as ti

# 初始化 Taichi，設定運算後端為 GPU (自動選擇 Metal, CUDA 或 Vulkan)
ti.init(arch=ti.gpu)

# 定義視窗解析度
res_x, res_y = 800, 600
# 建立一個 2D Vector 場，用來存儲每個像素的 RGB 顏色 (3個 float32)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 定義可以從 GUI 調整的全局變數 (ti.field 允許 CPU 與 GPU 通訊)
light_pos_x = ti.field(ti.f32, shape=()) # 光源 X 座標
light_pos_y = ti.field(ti.f32, shape=()) # 光源 Y 座標
light_pos_z = ti.field(ti.f32, shape=()) # 光源 Z 座標
box_pos_x = ti.field(ti.f32, shape=())   # 長方體中心 X 座標
box_pos_z = ti.field(ti.f32, shape=())   # 長方體中心 Z 座標
max_bounces = ti.field(ti.i32, shape=()) # 光線最大反彈次數 (控制反射深度)

# 材質類型的枚舉定義
MAT_DIFFUSE = 0  # 漫反射材質 (如地板、紅球、長方體)
MAT_MIRROR = 1   # 鏡面材質 (如銀球)

@ti.func
def normalize(v):
    """將向量歸一化，使其長度為 1 (防止除以 0 加入 1e-5)"""
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    """計算反射向量：入射方向 I 在法線 N 上的完美鏡面反射"""
    return I - 2.0 * I.dot(N) * N

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """球體求交：回傳 (撞擊距離 t, 撞擊點法線)"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center # 圓心到光線起點的向量
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c # 二次方程判別式
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0 # 計算近處的交點
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center) # 球體法線是從中心指向撞擊點
    return t, normal

@ti.func
def intersect_aabb(ro, rd, box_min, box_max):
    """長方體求交 (Slab Method)：檢查光線是否穿過三個軸向的重疊區間"""
    t_near = -1e10 # 進入盒子的最近距離
    t_far = 1e10  # 離開盒子的最遠距離
    normal = ti.Vector([0.0, 0.0, 0.0])
    is_hit = 1 

    for i in ti.static(range(3)): # 靜態循環，編譯時展開處理 X, Y, Z 三軸
        if ti.abs(rd[i]) < 1e-6: # 若光線平行於此軸
            if ro[i] < box_min[i] or ro[i] > box_max[i]: # 且起點不在範圍內則不相交
                is_hit = 0
        else:
            t1 = (box_min[i] - ro[i]) / rd[i] # 此軸入射時間
            t2 = (box_max[i] - ro[i]) / rd[i] # 此軸出射時間
            tn = ti.min(t1, t2)
            tf = ti.max(t1, t2)
            
            if tn > t_near: # 更新全局進入時間，並根據當前軸向決定法線
                t_near = tn
                n = ti.Vector([0.0, 0.0, 0.0])
                if i == 0: n = ti.Vector([-1.0, 0.0, 0.0]) if rd[0] > 0 else ti.Vector([1.0, 0.0, 0.0])
                if i == 1: n = ti.Vector([0.0, -1.0, 0.0]) if rd[1] > 0 else ti.Vector([0.0, 1.0, 0.0])
                if i == 2: n = ti.Vector([0.0, 0.0, -1.0]) if rd[2] > 0 else ti.Vector([0.0, 0.0, 1.0])
                normal = n
            if tf < t_far: # 更新全局離開時間
                t_far = tf

    t_res = -1.0
    # 若進入時間小於離開時間且離開時間在前方，則視為撞擊
    if is_hit == 1 and t_near <= t_far and t_far >= 0:
        t_res = t_near
    else:
        normal = ti.Vector([0.0, 0.0, 0.0])
    return t_res, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    """無限大平面求交：計算光線何時到達指定的 Y 高度"""
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5: # 避免除以零
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    """遍歷場景中所有物體，尋找最近的交點並回傳其屬性"""
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 檢測綠色長方體 (動態位置)
    bx, bz = box_pos_x[None], box_pos_z[None]
    # box_min/max 定義長方體的空間包圍盒
    box_min = ti.Vector([bx - 1.0, -2.1, bz - 0.5]) 
    box_max = ti.Vector([bx + 1.0, -1.1, bz + 0.5])
    t, n = intersect_aabb(ro, rd, box_min, box_max)
    if 0 < t < min_t:
        min_t, hit_n, hit_c, hit_mat = t, n, ti.Vector([0.1, 0.8, 0.1]), MAT_DIFFUSE

    # 2. 檢測紅色漫反射球 (固定位置)
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t, hit_n, hit_c, hit_mat = t, n, ti.Vector([0.8, 0.1, 0.1]), MAT_DIFFUSE

    # 3. 檢測銀色鏡面球 (固定位置)
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t, hit_n, hit_c, hit_mat = t, n, ti.Vector([0.9, 0.9, 0.9]), MAT_MIRROR

    # 4. 檢測棋盤格地板
    t, n = intersect_plane(ro, rd, -2.5)
    if 0 < t < min_t:
        min_t, hit_n, hit_mat = t, n, MAT_DIFFUSE
        p = ro + rd * t
        # 使用地板座標計算棋盤格顏色
        grid_scale = 2.0
        ix, iz = ti.floor(p.x * grid_scale), ti.floor(p.z * grid_scale)
        hit_c = ti.Vector([0.3, 0.3, 0.3]) if (ix + iz) % 2 == 0 else ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

@ti.kernel
def render():
    """渲染核心：計算每個像素的光線路徑"""
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2]) # 設定深藍色背景

    for i, j in pixels: # 自動並行化：每個像素同時執行
        # 屏幕座標映射到世界空間的視野方向
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0
        ro = ti.Vector([0.0, 1.5, 7.0]) # 相機位置
        rd = normalize(ti.Vector([u, v - 0.3, -1.0])) # 相機前進方向

        final_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0]) # 光線剩餘能量權重
        
        # 光線反彈循環
        for bounce in range(max_bounces[None]):
            t, N, obj_color, mat_id = scene_intersect(ro, rd)
            if t > 1e9: # 若未擊中物體，加上背景色後終止
                final_color += throughput * bg_color
                break
                
            p = ro + rd * t # 計算交點座標
            if mat_id == MAT_MIRROR: # 鏡面反射處理
                ro = p + N * 1e-4 # 從交點微移發射新光線 (防自相交)
                rd = normalize(reflect(rd, N))
                throughput *= 0.8 * obj_color # 反射損耗能量
            elif mat_id == MAT_DIFFUSE: # 漫反射處理
                L = normalize(light_pos - p) # 交點往光源的方向
                # 陰影射線檢測
                shadow_ray_orig = p + N * 1e-4
                shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
                dist_to_light = (light_pos - p).norm()
                
                direct_light = 0.2 * obj_color # 基礎環境光
                if shadow_t >= dist_to_light: # 若無遮擋則計算 Lambertian 漫反射
                    diff = ti.max(0.0, N.dot(L))
                    direct_light += 0.8 * diff * obj_color
                
                final_color += throughput * direct_light # 累加顏色
                break # 漫反射後光線視為被吸收或隨機散射，結束追蹤

        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0) # 限制顏色在有效範圍內

def main():
    # 建立 GGUI 視窗
    window = ti.ui.Window("Ray Tracing - Green Box Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 設置預設參數
    light_pos_x[None], light_pos_y[None], light_pos_z[None] = 2.0, 6.0, 3.0
    box_pos_x[None], box_pos_z[None] = 1.2, 0.0
    max_bounces[None] = 4

    while window.running:
        render() # 呼叫 GPU 執行渲染
        canvas.set_image(pixels) # 將結果畫面上傳到視窗
        # 繪製互動式 GUI 控制面板
        with gui.sub_window("Controls", 0.6, 0.05, 0.35, 0.35):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 10.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            box_pos_x[None] = gui.slider_float('Box X Pos', box_pos_x[None], -3.0, 3.0)
            box_pos_z[None] = gui.slider_float('Box Z Pos', box_pos_z[None], -3.0, 3.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 5)
        window.show()

if __name__ == '__main__':
    main()