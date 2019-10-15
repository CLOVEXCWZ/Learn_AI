<font  size=15>**MarkDown 公式**</font>



[TOC]

## 1 数学公式

​	主要编辑一些常用的数学公式以及符号。
### 1.1 基本公式


|     描述     |                   公式                    |
|:----------: | :---------------------------------------: |
|     上标     |   x^2                    |
|     下标     |                    x_1|
|     分式     |                \frac{1}{3}                |
|    省略号    | \cdots                   |
|    开根号    |                 \sqrt{2}|
|     矢量     |                  \vec{a}                  |
|     积分     |\int{x}dx                 |
|     极限     | \int{1}^{2}{x}dx  lim{n\rightarrow+\infty} |
|     累加     |        \sum{a} \sum_{n=1}^{100}a_n|
|     累乘     |      \prod{x}  \prod_{n=1}^{99}{x_n}      |
|    平均值    | \overline{a}                |
| 线性回归 y尖 |                \widehat{a} |
|  等价无穷小  |               \widetilde{-}               |
|   一阶导数   | \dot{}                   |
|   二阶导数   |                  \ddot{}|




$$
上标: P^i \\
下标: P_i  \\
分式:\frac{1}{2}\\
省略号: \cdots \\
开根号: \sqrt{2}\\
矢量:\vec{a}\\
积分: \int{x}dx\\
极限1:\lim{a+b}\\
极限2:\lim_{n\rightarrow+\infty}\\
累加1:\sum{a}\\
累加2:\sum_{i=1}^{100}{n_i} \\
累乘1: \prod{x} \\
累乘2：\prod_{i=1}^{10}{x_i}\\
平均值：\overline{a}\\
y预测值:\widehat{a}\\
等价无穷小：\widetilde{-}\\
一阶导数：a\dot{}\\
二阶导数：a\ddot{}
$$

### 1.2 希腊字母

|      |             |
| :--: | :------: |
| *A*  |    A |
| *α*  |   \alpha    |
| *B*  |    B     |
| *β*  |    \beta    |
|  Γ   |  \Gamma |
| *γ*  |   \gamma    |
|  Δ   |  \Delta  |
| *δ*  |   \delta    |
| *E*  |    E |
| *ϵ*  |  \epslion   |
| *ε*  | \varepslion |
| *Z*  |    Z|
| *ζ*  |    \zeta    |
| *H*  |    H     |
| *η*  |    \eta     |
|  Θ   |  \Theta|
| *θ*  |   \theta    |
| *I*  |    I     |
| *ι*  |    \iota    |
| *K*  |    K|
| *κ*  |   \kappa    |
|  Λ   | \Lambda  |
| *λ*  |   \lambda   |
| *M*  |    M|
| *μ*  |     \mu     |
| *N*  |    N     |
| *ν*  |     \nu     |
|  Ξ   |   \Xi|
| *ξ*  |     \xi     |
| *O*  |    O   |
| *ο*  |  \omicron   |
|  Π   |   \pi|
| *π*  |     \pi     |
| *P*  |    p     |
| *ρ*  |    \rho     |
|  Σ   |  \sigma|
| *σ*  |   \sigma    |
| *T*  |    T    |
| *τ*  |    \tau     |
|  Υ   | \Upslion |
| *υ*  |  \upslion   |
|  Φ   |   \Phi   |
| *ϕ*  |    \phi     |
| 导数 | \partial |
| *φ*  |   \varphi   |
| *X*  |    X     |
| *χ*  |    \chi     |
| *ψ*  |    \psi     |
|  Ψ   |   \Psi   |
| *ω*  |   \omega    |
|  Ω |  \Pmega  |

### 1.3 三角函数

- sin -> \sin

$$
\sin{A}\\
\cos{B}\\
\tan{C}\\
\cot{D}
$$

### 1.4 关系运算符

|      描述      |  公式   |
|:------------: | :-----: |
| e为底数的对数  |  \ln2   |
|      对数      | \log_28 |
| 10为底数的对数 |  \lg10  |

$$
e为底数的对数:\ln2 \\
对数: \log_28 \\
10为底数的对数: \lg10
$$
### 1.5 关系运算符

|   描述   |    公式    |
| :------: | :--------: |
|    +-    | \pm     |
|    点    |   \cdot    |
|    除    |    \div    |
| 大于等于 |    \leq    |
| 小于等于 |    \geq    |
|    ∀     |  \forall   |
|    ∞     |   \infty   |
|    ∅     | \emptyset  |
|    ∃     |  \exists   |
|    ∇     | \nabla   |
|    ⊥     |    \bot    |
|    ∠     |   \angle   |
|    ∵     |\because  |
|    ∴     | \therefore |

$$
+-:\pm\\
点:\cdot\\
除:\div\\
大于等于:\leq\\
小于等于:\geq\\
∀:\forall\\
∞:\infty\\
∅:\emptyset\\
∃:\exists\\
∇:\nabla\\
⊥:\bot\\
∠:\angle\\
∵:\because\\
∴:\therefore
$$



### 1.6 行列式矩阵

```{.python .input}
"""
=========== 行列式
\left\|
\begin{matrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{matrix} 
\right\| \tag{1}

============= 矩阵
\left[
 \begin{matrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{matrix}
  \right] \tag{2}
  
============== 矩阵
\left[
\begin{matrix}
 1      & 2      & \cdots & 4      \\
 7      & 6      & \cdots & 5      \\
 \vdots & \vdots & \ddots & \vdots \\
 8      & 9      & \cdots & 0      \\
\end{matrix}
\right] \tag{3}

================ 大括号
\left\{
\begin{matrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{matrix} 
\right\} \tag{4}

"""
```

$$
\left\|
\begin{matrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{matrix}
\right\| \tag{1}   
$$

$$
\left[
 \begin{matrix}
   1 & 2 & 3 \\
   4 & 5 & 6
\\
   7 & 8 & 9
  \end{matrix}
  \right] \tag{2}
$$

$$
\left[
\begin{matrix}
 1
& 2      & \cdots & 4      \\
 7      & 6      & \cdots & 5      \\
 \vdots &
\vdots & \ddots & \vdots \\
 8      & 9      & \cdots & 0      \\
\end{matrix}
\right] \tag{3}
$$

$$
\left\{
\begin{matrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 &
9
\end{matrix} 
\right\} \tag{4}
$$





### 1.7 大括号

```{.python .input}
"""
=================== 1
f(x)=\left\{
\begin{aligned}
x & = & \cos(t) \\
y & = & \sin(t) \\
z & = & \frac xy
\end{aligned}
\right.  \tag{1}

==================== 2
F^{HLLC}=\left\{
\begin{array}{rcl}
F_L       &      & {0      <      S_L}\\
F^*_L     &      & {S_L \leq 0 < S_M}\\
F^*_R     &      & {S_M \leq 0 < S_R}\\
F_R       &      & {S_R \leq 0}
\end{array} \right. \tag{2}

===================== 3
f(x)=
\begin{cases}
0& \text{x=0}\\
1& \text{x!=0}
\end{cases}   \tag{3}

"""
```

$$
f(x)=\left\{
\begin{aligned}
x & = & \cos(t) \\
y & = & \sin(t) \\
z & = &
\frac xy
\end{aligned}
\right. \tag{1}
$$

$$
F^{HLLC}=\left\{
\begin{array}{rcl}
F_L       &      & {0      <      S_L}\\
F^*_L     &      &
{S_L \leq 0 < S_M}\\
F^*_R     &      & {S_M \leq 0 < S_R}\\
F_R       &      &
{S_R \leq 0}
\end{array} \right. \tag{2}
$$

$$
f(x)=
\begin{cases}
0&
\text{x=0}\\
1& \text{x!=0}
\end{cases}   \tag{3}
$$
