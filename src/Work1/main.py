# src/Work1/main.py


# 注意：初始化必须在最前面执行，接管底层 GPU
import taichi as ti
import math

# 初始化 Taichi
ti.init(arch=ti.cpu)

# ---------------------------------------------------------
# 新增：純 Python 範圍可用的向量輔助函數
# ---------------------------------------------------------
def vec_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

def vec_normalize(v):
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length == 0:
        return [0.0, 0.0, 0.0]
    return [v[0]/length, v[1]/length, v[2]/length]

def vec_cross(a, b):
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]

# ---------------------------------------------------------
# 1. 矩陣生成函數
# ---------------------------------------------------------
def get_model_matrix(translation, rotation_z_angle, rotation_y_angle):
    rad_z = math.radians(rotation_z_angle)
    cos_z, sin_z = math.cos(rad_z), math.sin(rad_z)
    R_z = ti.Matrix([
        [cos_z, -sin_z, 0, 0],
        [sin_z,  cos_z, 0, 0],
        [    0,      0, 1, 0],
        [    0,      0, 0, 1]
    ])
    
    rad_y = math.radians(rotation_y_angle)
    cos_y, sin_y = math.cos(rad_y), math.sin(rad_y)
    R_y = ti.Matrix([
        [ cos_y, 0, sin_y, 0],
        [     0, 1,     0, 0],
        [-sin_y, 0, cos_y, 0],
        [     0, 0,     0, 1]
    ])
    
    T = ti.Matrix([
        [1, 0, 0, translation[0]],
        [0, 1, 0, translation[1]],
        [0, 0, 1, translation[2]],
        [0, 0, 0, 1]
    ])
    return T @ R_y @ R_z

def get_view_matrix(eye, target, up):
    # 使用我們自訂的純 Python 函數來計算向量
    z_dir = vec_sub(eye, target)
    z_axis = vec_normalize(z_dir)
    
    x_dir = vec_cross(up, z_axis)
    x_axis = vec_normalize(x_dir)
    
    y_axis = vec_cross(z_axis, x_axis)
    
    R = ti.Matrix([
        [x_axis[0], x_axis[1], x_axis[2], 0],
        [y_axis[0], y_axis[1], y_axis[2], 0],
        [z_axis[0], z_axis[1], z_axis[2], 0],
        [0,         0,         0,         1]
    ])
    T = ti.Matrix([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
        [0, 0, 0, 1]
    ])
    return R @ T

def get_projection_matrix(fov_degrees, aspect_ratio, near, far):
    fov_rad = math.radians(fov_degrees)
    f = 1.0 / math.tan(fov_rad / 2.0)
    P = ti.Matrix([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0] 
    ])
    return P

# ---------------------------------------------------------
# 2. 視窗渲染與互動控制迴圈
# ---------------------------------------------------------
def main():
    res_x, res_y = 700, 700
    gui = ti.GUI("3D Triangle - Fixed Version", res=(res_x, res_y))

    vertices_local = [
        ti.Vector([ 2.0,  0.0, -2.0, 1.0]), # v_1
        ti.Vector([ 0.0,  2.0, -2.0, 1.0]), # v_2
        ti.Vector([-2.0,  0.0, -2.0, 1.0])  # v_3
    ]

    angle_z = 0.0
    angle_y = 0.0 
    rotation_speed_z = 2.0 
    rotation_step_y = 15.0 

    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.LMB:
                angle_y -= rotation_step_y  
        
        if gui.is_pressed('a', 'A'):
            angle_z -= rotation_speed_z 
        if gui.is_pressed('d', 'D'):
            angle_z += rotation_speed_z 

        mouse_x, mouse_y = gui.get_cursor_pos()
        translate_x = (mouse_x - 0.5) * 12.0
        translate_y = (mouse_y - 0.5) * 12.0
        translate_z = 0.0 

        gui.clear(0x222222)

        M = get_model_matrix(
            translation=[translate_x, translate_y, translate_z], 
            rotation_z_angle=angle_z,
            rotation_y_angle=angle_y
        )
        V = get_view_matrix(eye=[0, 3, 8], target=[0, 0, 0], up=[0, 1, 0])
        P = get_projection_matrix(fov_degrees=45, aspect_ratio=res_x/res_y, near=0.1, far=100.0)
        
        MVP = P @ V @ M

        screen_coords = []
        for v_local in vertices_local:
            v_clip = MVP @ v_local
            w = v_clip[3]
            v_ndc = v_clip / w if w != 0 else v_clip
            
            x_screen = (v_ndc[0] + 1.0) / 2.0
            y_screen = (v_ndc[1] + 1.0) / 2.0
            screen_coords.append([x_screen, y_screen])

        p1, p2, p3 = screen_coords[0], screen_coords[1], screen_coords[2]
        
        gui.line(p1, p2, color=0xFF5555, radius=3)
        gui.line(p2, p3, color=0x55FF55, radius=3)
        gui.line(p3, p1, color=0x5555FF, radius=3)
        
        gui.circles(ti.Vector([p1, p2, p3]).to_numpy(), color=0xFFFFFF, radius=4)

        gui.text("Move Mouse to Translate", pos=(0.05, 0.95), color=0xFFFFFF)
        gui.text("Left Click: Rotate Y-axis Clockwise", pos=(0.05, 0.90), color=0xFFFFFF)
        gui.text("Hold 'A' / 'D': Rotate Z-axis", pos=(0.05, 0.85), color=0xFFFFFF)
        gui.text("Press 'ESC' to exit", pos=(0.05, 0.80), color=0xFFFFFF)
        
        gui.show()

if __name__ == "__main__":
    main()