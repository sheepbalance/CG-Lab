# src/Work3/main.py
import taichi as ti

# 1. 初始化 Taichi，預設會根據你的硬體選擇 Vulkan 或 x64
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 常數定義
INF = 1e10
camera_pos = ti.Vector([0.0, 0.0, 5.0])
light_pos = ti.Vector([2.0, 3.0, 4.0])
light_color = ti.Vector([1.0, 1.0, 1.0])
bg_color = ti.Vector([0.0, 0.2, 0.2])

# UI 參數場 (使用單一元素的 field 以便與 UI 綁定)
ka = ti.field(ti.f32, shape=())
kd = ti.field(ti.f32, shape=())
ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())

# 初始化預設值
ka[None], kd[None], ks[None], shininess[None] = 0.2, 0.7, 0.5, 32.0

@ti.func
def intersect_sphere(pos, dir, center, radius):
    to_obj = pos - center
    a = dir.dot(dir)
    b = 2.0 * to_obj.dot(dir)
    c = to_obj.dot(to_obj) - radius**2
    det = b**2 - 4.0 * a * c
    t, normal = INF, ti.Vector([0.0, 0.0, 0.0])
    if det >= 0:
        t_near = (-b - ti.sqrt(det)) / (2.0 * a)
        if t_near > 0:
            t = t_near
            normal = ((pos + t * dir) - center).normalized()
    return t, normal

@ti.func
def intersect_cone(pos, dir, apex, bottom_y, radius):
    h = apex.y - bottom_y
    m = (radius / h)**2
    O, D = pos - apex, dir
    a = D.x**2 + D.z**2 - m * D.y**2
    b = 2.0 * (O.x * D.x + O.z * D.z - m * O.y * D.y)
    c = O.x**2 + O.z**2 - m * O.y**2
    det = b**2 - 4.0 * a * c
    t, normal = INF, ti.Vector([0.0, 0.0, 0.0])
    
    if det >= 0:
        sqrt_det = ti.sqrt(det)
        t0 = (-b - sqrt_det) / (2.0 * a)
        t1 = (-b + sqrt_det) / (2.0 * a)
        
        # 手動拆解判斷，完全不使用迴圈
        # 檢查第一個交點 t0
        if t0 > 0 and t0 < t:
            hit_p = pos + t0 * dir
            if bottom_y <= hit_p.y <= apex.y:
                t = t0
                cp = hit_p - apex
                r_vec = ti.Vector([cp.x, 0, cp.z])
                normal = (r_vec.normalized() * (h/radius) + ti.Vector([0, radius/h, 0])).normalized()
        
        # 檢查第二個交點 t1
        if t1 > 0 and t1 < t:
            hit_p = pos + t1 * dir
            if bottom_y <= hit_p.y <= apex.y:
                t = t1
                cp = hit_p - apex
                r_vec = ti.Vector([cp.x, 0, cp.z])
                normal = (r_vec.normalized() * (h/radius) + ti.Vector([0, radius/h, 0])).normalized()
    
    # 底部圓蓋交點
    t_plane = (bottom_y - pos.y) / dir.y
    if 0 < t_plane < t:
        hit_p = pos + t_plane * dir
        if (hit_p.x - apex.x)**2 + (hit_p.z - apex.z)**2 <= radius**2:
            t, normal = t_plane, ti.Vector([0.0, -1.0, 0.0])
            
    return t, normal

@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x / 2) / (res_y / 2)
        v = (j - res_y / 2) / (res_y / 2)
        ray_dir = ti.Vector([u, v, -1.0]).normalized()

        # 物體屬性
        s_c, s_r, s_col = ti.Vector([-1.2, -0.2, 0.0]), 1.2, ti.Vector([0.8, 0.1, 0.1])
        c_a, c_b_y, c_r, c_col = ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2, ti.Vector([0.6, 0.2, 0.8])
        
        # 執行光線求交
        t_s, n_s = intersect_sphere(camera_pos, ray_dir, s_c, s_r)
        t_c, n_c = intersect_cone(camera_pos, ray_dir, c_a, c_b_y, c_r)
        
        # 深度測試 (Z-Buffer Logic)
        hit_t, N, base_col = INF, ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        if t_s < t_c:
            hit_t, N, base_col = t_s, n_s, s_col
        elif t_c < INF:
            hit_t, N, base_col = t_c, n_c, c_col

        if hit_t < INF:
            # Phong Shading
            P = camera_pos + hit_t * ray_dir
            L = (light_pos - P).normalized()
            V = (camera_pos - P).normalized()
            R = (-L + 2.0 * N.dot(L) * N).normalized()
            
            amb = ka[None] * base_col
            diff = kd[None] * ti.max(N.dot(L), 0.0) * base_col
            spec = ks[None] * ti.pow(ti.max(R.dot(V), 0.0), shininess[None]) * light_color
            
            pixels[i, j] = amb + diff + spec
        else:
            pixels[i, j] = bg_color

def main():
    window = ti.ui.Window("Taichi Phong RayCaster", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Shader Settings", x=0.55, y=0.02, width=0.40, height=0.3):
            gui.text("Phong Parameters")
            
            # 修正：手動讀取與存回 Field 的值
            # 傳入 ka[None] (數值)，並將結果賦回 ka[None]
            ka[None] = gui.slider_float("Ka (Ambient)", ka[None], 0.0, 1.0)
            kd[None] = gui.slider_float("Kd (Diffuse)", kd[None], 0.0, 1.0)
            ks[None] = gui.slider_float("Ks (Specular)", ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float("Shininess", shininess[None], 1.0, 128.0)
        
        window.show()

if __name__ == '__main__':
    main()