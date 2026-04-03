# AND2_X1_ML — YMTC CJ3 工艺布局理解

## 1. 电路功能

AND2 = NAND2(A,B) → net2 → INV(net2) → Y

```
A ─┬─[PMOS T0 P]─┬── net2 ──[PMOS T3]──── Y
   │             │           (G=net2)
B ─┼─[PMOS T1 P]─┘
   │                [DUMMY T2 P: VDD-VDD]
   │
A ─┤─[NMOS T0 N ser]─ net0 ─[NMOS T1 N ser]── VSS
B ─┘               (G=A)                   (G=B)

                [DUMMY T2 N: VSS-VSS]

           Y ──[NMOS T3 N]── VSS
              (G=net2, short, y=11..14 only)
```

## 2. 晶体管拓扑（来自 AND2_X1_ML.xlsx）

### xlsx 坐标系

- 列方向（x 轴）：表格头行，x = 0..8
- 行方向（y 轴）：表格 A 列，y = 0..17
- 偶数列 x = 0,2,4,6,8：Source/Drain 扩散列
- 奇数列 x = 1,3,5,7：Gate 多晶硅列

### 有源区数据

| y 区间 | 区域 | x=0 | x=1 | x=2 | x=3 | x=4 | x=5 | x=6 | x=7 | x=8 |
|--------|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 1..5   | PMOS | VDD | A   | net2| B   | VDD | **None**| VDD | net2| Y  |
| 6..10  | 通道 | -   | A   | -   | B   | -   | -   | -   | net2| -  |
| 11..16 | NMOS(左)| net2| A | net0| B | VSS | **None**| — | — | — |
| 11..14 | NMOS(右)| —  | —  | —   | —  | —   | —   | VSS | net2| Y |
| 0, 17  | 边界 | —  | —   | —   | —  | —   | —   | —  | —   | — |

> **注**：y=0 和 y=17 为边界行，M0 不可布线。

### 晶体管识别（S-G-D 连续排列）

| 晶体管 | x 列 | Gate(x) | PMOS S/D         | NMOS S/D         | 类型   |
|--------|------|---------|------------------|------------------|--------|
| T0     | 0-1-2 | A(x=1)  | S=VDD(x=0), D=net2(x=2) | S=net2(x=0), D=net0(x=2) | NAND2 |
| T1     | 2-3-4 | B(x=3)  | S=net2(x=2), D=VDD(x=4) | S=net0(x=2), D=VSS(x=4) | NAND2 |
| T2     | 4-5-6 | None(x=5)| S=D=VDD | S=D=VSS | **DUMMY** |
| T3     | 6-7-8 | net2(x=7)| S=VDD(x=6), D=Y(x=8) | S=VSS(x=6), D=Y(x=8) | INV |

**Dummy 判断**：T2 的 gate=None 且 S==D（PMOS:VDD=VDD, NMOS:VSS=VSS）→ DUMMY，x=5 列无任何布线。

**晶体管排列规则**：前一个晶体管的 D 与后一个晶体管的 S 共享同一列（无 L/R active 分组概念）。

### NMOS 有源区高度差异

- T0, T1 NMOS（x=0,2,4）：高度 6，y=11..16
- T3 NMOS（x=6,8）：高度 4，y=11..14（对应 T3 晶体管尺寸较小）

---

## 3. 网格规格

```
COLS = 9   (x = 0..8)
ROWS = 18  (y = 0..17)
层数 = 3   (M0, M1, M2)
```

### 行分区

| y 范围 | 区域          | M0 约束            | M1 约束          |
|--------|---------------|-------------------|-----------------|
| y=0    | VDD power rail | 无 M0 节点         | 全连接水平 rail   |
| y=1..5 | PMOS 有源区    | SD: 无边（via-only）| 正常布线         |
| y=6..10| 布线通道       | Gate: 竖向+via     | 正常布线         |
| y=11..16| NMOS 有源区  | SD: 无边（via-only）| 正常布线         |
| y=17   | VSS power rail | 无 M0 节点         | 全连接水平 rail   |

---

## 4. YMTC CJ3 工艺规则（在此单元的应用）

### 4.1 S/D 列 M0 规则
- **无任何 M0 横纵连线**（x = 0,2,4,6,8）
- SD 通过 M0↔M1 via 接入路由，但 via 只在有源区边界：
  - PMOS 侧：y=1（有源区最顶端，靠近 VDD rail）
  - NMOS 侧（高 6）：y=16（有源区最底端，靠近 VSS rail），适用于 x=0,2,4
  - NMOS 侧（高 4）：y=14（有源区最底端），适用于 x=6,8

### 4.2 Gate 列 M0 规则
- Gate 列（x=1,3,7）可竖向布线（y=1..16）
- **M0↔M1 via 只允许在非有源区（y=6..10）**
- Gate 不能在 M0 直接横向连接 S/D 列
- x=5（Dummy gate）：无任何 M0 边，无 via

### 4.3 M1 电源轨规则
- y=0：VDD power rail（全行所有横向边必须相连）
- y=17：VSS power rail（同上）
- M0 y=0 和 y=17 无节点（不可布线）
- VDD/VSS 在 M1 层只能纵向连接到 power rail，不能横向连接至其他 VDD/VSS
- **优化**：M1 y=0↔y=1 和 y=16↔y=17 的纵向边代价设为 0（鼓励直连电源轨）

### 4.4 M2 规则
- **仅 y=7 和 y=9** 可布线（两行横向）
- 只允许横向走线，无 y 方向直连
- 不用于连接 VDD/VSS

### 4.5 Active 覆盖约束（ActiveOccupancyConstraint）
每个 S/D Terminal 在 M1 层对应列、对应 y 范围内必须覆盖全部有源区节点数：

| 线网  | 列 x | 行类型      | y 范围 | 要求节点数 |
|-------|------|-------------|--------|-----------|
| VDD   | 0    | PMOS        | 1..5   | 5         |
| VDD   | 4    | PMOS        | 1..5   | 5         |
| VDD   | 6    | PMOS        | 1..5   | 5         |
| net2  | 2    | PMOS        | 1..5   | 5         |
| net2  | 0    | NMOS H6     | 11..16 | 6         |
| net0  | 2    | NMOS H6     | 11..16 | 6         |
| VSS   | 4    | NMOS H6     | 11..16 | 6         |
| VSS   | 6    | NMOS H4     | 11..14 | 4         |
| net_Y | 8    | PMOS        | 1..5   | 5         |
| net_Y | 8    | NMOS H4     | 11..14 | 4         |

### 4.6 Pin 引出规则
- net_A（输入 A）、net_B（输入 B）、net_Y（输出 Y）需在 M1 层引出 pin
- VDD、VSS 无需 pin 引出

---

## 5. 线网定义

| 线网  | 终端 (EXT/M0/M1)                         | 说明                     |
|-------|------------------------------------------|--------------------------|
| VDD   | EXT(0,PMOS), EXT(4,PMOS), EXT(6,PMOS), M1(0,0), M1(4,0), M1(6,0) | PMOS 电源 |
| VSS   | EXT(4,H6), EXT(6,H4), M1(4,17), M1(6,17) | NMOS 地                  |
| net_A | M0(1,1), M0(1,16), PinSpec(M1)          | 共栅 A，需 M1 pin         |
| net_B | M0(3,1), M0(3,16), PinSpec(M1)          | 共栅 B，需 M1 pin         |
| net2  | EXT(2,PMOS), EXT(0,H6), M0(7,1)         | NAND2 输出/INV 输入       |
| net0  | EXT(2,H6), M1(2,11)                      | NMOS 内部串联节点（单 SD + 锚点）|
| net_Y | EXT(8,PMOS), EXT(8,H4), PinSpec(M1)     | AND2 输出，需 M1 pin      |

> EXT(x, ROW_TYPE) 是虚拟节点，通过零代价边连接到 M0(x, boundary_y)。

---

## 6. 行类型定义

```python
PMOS_ROW    = 0   # PMOS，y=1..5
NMOS_H6_ROW = 1   # NMOS 高度 6，y=11..16（x=0,2,4）
NMOS_H4_ROW = 2   # NMOS 高度 4，y=11..14（x=6,8）
```

---

## 7. 关键设计决策

1. **EXT 虚拟节点**：代表 SD 端点的"外部起点"，零代价连到唯一的 M0 有源区边界点。注册为虚拟层（不参与 space 约束、不计入 active 覆盖节点数）。

2. **Gate via 限制**：M0↔M1 via 仅在 y=6..10（非有源区），防止通孔损伤有源区，Gate 通过 M0 竖向走线到通道区再上 M1。

3. **net0 单 SD 终端问题**：net0 只有一个 EXT 终端（NMOS x=2，需要 active 覆盖 y=11..16），通过添加 M1(2,11) 作为第二锚点触发 Steiner 布线，使 M1 覆盖完整 active 区域。

4. **M2 仅两行**：y=7 和 y=9（不是整个通道 y=6..10），这是工艺 DRC 规则限制。

5. **Power rail 零代价**：M1 y=0↔y=1 和 y=16↔y=17 的纵向边代价设为 0，优先引导 VDD/VSS 垂直到电源轨而非水平绕行。
