# Jensen 不等式

- 凸函数

$$
\lambda f(x_1)+(1-\lambda)f(x_2)\geq f(\lambda x_1 + (1-\lambda)x_2)
$$

- 于是

$$
对于任意x_i，若\lambda_i\geq0 且\\
\sum_{i}\lambda_i=1
$$

​	使用数学归纳法，可以证明凸函数f(x)满足：
$$
f(\sum_{i=1}^{M}\lambda_ix_i)\leq\sum_{i=1}^{M}\lambda_if(x_i)
$$
如果把 ![\lambda_i](https://www.zhihu.com/equation?tex=%5Clambda_i) 看成取值为 ![{x_i}](https://www.zhihu.com/equation?tex=%7Bx_i%7D) 的离散变量 x 的概率分布，那么公式(2)就可以写成 
$$
f(E[x])\leq E[f(x)]
$$


