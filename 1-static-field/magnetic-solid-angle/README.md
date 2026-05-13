# 立体角（Solid Angle）磁标势法：任意形状电流回路的磁场

本目录给出一个数值示例：对位于同一平面内的闭合电流回路（方形、圆形、任意形状），用“立体角磁标势”计算磁标势并由梯度得到磁感应强度分布；同时给出圆形与方形在线圈中心轴线上方的解析结果用于对比。

---

## 1. 方法要点

在无自由电流区域（观测点不在电流线上，且不穿过选取的割面），有

$$\nabla\times \vec{H}=0,\qquad \vec{H}=-\nabla \varphi_m,\qquad \vec{B}=\mu_0\vec{H}.$$

对一条闭合电流回路 $C$，任选一张以 $C$ 为边界的曲面 $\Sigma$ 作为“割面”。在割面同侧的无电流区域，可以用立体角表示磁标势：

$$\boxed{\ \varphi_m(\mathbf{r})=\frac{I}{4\pi}\,\Omega(\mathbf{r})\ }\qquad(\mathbf{r}\notin \Sigma)$$

其中 $\Omega(\mathbf{r})$ 是曲面 $\Sigma$ 在观测点 $\mathbf{r}$ 处张成的**有向立体角**。跨过割面时 $\Omega$ 会发生 $4\pi$ 的跳变，因此 $\varphi_m$ 在割面两侧相差 $I$，这正对应电流回路的“多值势/割面势”构造。

磁场由梯度给出：

$$\boxed{\ \vec{B}(\mathbf{r})=-\mu_0\nabla \varphi_m(\mathbf{r})\ =-\mu_0\frac{I}{4\pi}\nabla\Omega(\mathbf{r})\ }.$$

本示例在固定平面 $z=z_0>0$ 上计算 $\varphi_m(x,y,z_0)$，用数值梯度得到 $B_x,B_y$；并用对 $z$ 的中心差分得到 $B_z$，从而给出 $z=z_0$ 平面上的磁场分布。

---

## 2. 数值实现（多边形立体角）

对任意平面线圈，将其离散为顶点序列 $\{\mathbf{v}_i\}$（按逆时针给出）。立体角用三角剖分累加：

$$\Omega(\mathbf{r})=\sum_{i=1}^{N-2}\Omega_\triangle(\mathbf{r};\mathbf{v}_0,\mathbf{v}_i,\mathbf{v}_{i+1}).$$

每个三角形的立体角使用 Oosterom–Strackee 公式（数值稳定）：

设 $\mathbf{r}_k=\mathbf{v}_k-\mathbf{r}$，则

$$
\Omega_\triangle
=2\arctan\frac{\mathbf{r}_1\cdot(\mathbf{r}_2\times \mathbf{r}_3)}
{\ |\mathbf{r}_1||\mathbf{r}_2||\mathbf{r}_3|
+(\mathbf{r}_1\cdot\mathbf{r}_2)|\mathbf{r}_3|
+(\mathbf{r}_2\cdot\mathbf{r}_3)|\mathbf{r}_1|
+(\mathbf{r}_3\cdot\mathbf{r}_1)|\mathbf{r}_2| }.
$$

---

## 3. 解析解（用于对比）

下面给出**轴线上方**（线圈中心正上方）$B_z$ 的解析结果，和脚本中数值结果（中心像素）对比。

### 3.1 圆形线圈（半径 $a$）

轴线上方距离为 $z$：

$$\boxed{\ B_z(0,0,z)=\frac{\mu_0 I a^2}{2\,(a^2+z^2)^{3/2}}\ }.$$

更一般的离轴解析表达式可用第一、第二类完全椭圆积分 $K(k),E(k)$ 写出（此处不展开）。

### 3.2 方形线圈（边长 $2a$）

轴线上方距离为 $z$（中心正上方）：

$$\boxed{\ B_z(0,0,z)=\frac{2\mu_0 I a^2}{\pi\,(a^2+z^2)\sqrt{2a^2+z^2}}\ }.$$

---

## 4. 运行方式

本目录拆分为 3 个脚本，分别对应：方形线圈、圆形线圈、任意形状线圈。每个脚本都会生成 $1\times 3$ 子图（$\varphi_m$ 等势线、$B_z$ 色图、平面内投影场线），并默认保存图片到当前目录。

运行（方形）：

```bash
python solid_angle_coil_square.py --no-show
```

运行（圆形）：

```bash
python solid_angle_coil_circle.py --no-show
```

运行（任意形状，使用脚本自带示例曲线）：

```bash
python solid_angle_coil_arbitrary.py --no-show
```

参数示例（通用）：

```bash
python solid_angle_coil_circle.py --I 1 --mu0 1 --z0 0.6 --radius 1.0 --grid 240 --lim 2.5 --no-show
```

任意形状也可以从顶点文件读取（两列 `x y` 或三列 `x y z`；支持 `.txt`/`.dat`/`.csv` 以及 `.npy`）：

```bash
python solid_angle_coil_arbitrary.py --vertices vertices.txt --no-show
```

默认输出文件名分别为 `solid_angle_coil_square.png`、`solid_angle_coil_circle.png`、`solid_angle_coil_arbitrary.png`。
