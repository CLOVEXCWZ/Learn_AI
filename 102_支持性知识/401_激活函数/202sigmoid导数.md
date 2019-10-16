<font size=15>**Sigmoid求导推导**</font>


$$
g(z)=\frac{1}{1+e^{-z}}
$$



$$
\frac{\partial g(z)}{\partial z}=\frac{\partial g(z)}{\partial{(1+e^{-z})}}\frac{\partial{(1+e^{-z})}}{e^{-z}}\frac{\partial{e^{-z}}}{-z}\frac{\partial{-z}}{z}\\
= \frac{1}{(1+e^{-z})^2}.1.-e^{-z}.(-1)\\
=\frac{e^{-z}}{(1+e^{-z})^2}\\
= \frac{1}{(1+e^{-z})}.\frac{e^{-z}}{(1+e^{-z})}\\
= \frac{1}{(1+e^{-z})}.\frac{(1-1)+e^{-z}}{(1+e^{-z})}\\
= \frac{1}{(1+e^{-z})}.\frac{(1+e^{-z})-1}{(1+e^{-z})}\\
=\frac{1}{(1+e^{-z})}(1-\frac{1}{(1+e^{-z})})\\
=g(x)(1-g(x))
$$
