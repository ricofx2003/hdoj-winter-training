---
title: hdoj刷题总结（1）
pin: false
author: ricofx
categories:
  - 思考
tags:
  - ACM
abbrlink: cd98
date: 2023-01-22 13:58:06
---

> 所有代码都在[https://github.com/ricofx2003/hdoj-winter-training](https://github.com/ricofx2003/hdoj-winter-training)，可以在 vjudge 搜索题目。

简单总结一下最近做过的 6 个专题练习：

1. 强连通分量
2. 差分约束，2sat
3. 树链剖分，线段树
4. 区间 dp
5. 树形 dp
6. 概率与期望

<!--more-->

## 1. 强连通分量

### 迷宫城堡

tarjan 处理强连通分量后，用并查集或者 dfs 简单判断是否连通。

### Summer Holiday

tarjan 处理强连通分量后，每个连通块只用通知电话费最低的人。

### Hawk-and-Chicken

tarjan 处理强连通分量后，简单计算支持人数

### Proving Equivalences

首先将已经等价的问题合并，会得到一个或者多个 DAG，因为要所有命题等价，显然不会存在入度或者出度为 0 的点。尽量将出度为 0 的点连另一个 DAG 入度为 0 的点，这样能形成一个环。 最后再把入度/出度仍然为 0 的点任意连边。

### The King’s Problem

tarjan 缩点后，得到 DAG。在一个州的点显然都在某一条路径上。那么就是求有向图的最小路径覆盖。可以参考[这篇文章](https://www.cnblogs.com/justPassBy/p/5369930.html)。

### Equivalent Sets

和 Proving Equivalences 是同一道题目。

### Intelligence System

tarjan 处理后，每个强连通分量只会被某个可以通知到它的点通知，而这个点不可能通知对方，那么这个点被通知的先后对连边到它的点没有影响，也就是每个强连通分量在入边中取边权最小的边做为得到通知的最小费用。

### 考研路茫茫——空调教室

> 众所周知，HDU的考研教室是没有空调的，于是就苦了不少不去图书馆的考研仔们。Lele也是其中一个。而某教室旁边又摆着两个未装上的空调，更是引起人们无限YY。

可以这很带专。
只用找到桥就行了，注意重边不可能成为桥。

## 差分约束，2sat

### Peaceful Commission

经典 2sat

### Building roads

很好的一道题。先二分答案，处理出哪些牛不能分别在两边，不能同时在某一边等等。然后 2sat 判断是否存在解。

### Get Luffy Out *

感觉题意不清楚。题意应该是如果一把钥匙使用后，那么和它配对的所有钥匙都不能使用，但是自身是可以继续用的。那么只用二分答案，根据充分条件跑 2sat 就行了。

### Intervals

选择的数设置为 1， 不选的设置为 0。区间的和可以转换为前缀和的差，问题就变成的典型的差分约束问题。

### Cashier Employment

这道题想把环变成链，然后套用上一道题的做法，但是并不成功。
每个时刻会被前 8 个小时开始上班的员工覆盖。其实也就是类似区间和的问题。
但是前 7 个小时很特殊，我们假设 sum[i] 表示前 i 个时刻开始上班的总人数。就有:

$$
sum[i] + sum[24] - sum[16 + i] >= R[i]
$$

然后我就想，是不是可以枚举 sum[24]，这样就和上一题一样了。但是仍然没有通过此题！因为我没有注意到，当解出 sum[24] 大于我们枚举时，并不代表就不存在 sum[24] 小于枚举的数的解。所以我们需要手动限制 sum[24] 的大小，看看是不是真的有解！

### King

也是对前缀和差分约束。

### Schedule Problem

对开始时间差分约束。

### Invitation Cards

每个学生单独付费，不存在搭顺风车。那么简单跑 n 次最短路就行了。。。

### XYZZY

显然用 spfa 跑最长路。需要注意的是有正环不一定就成功，还需要能从这个环走到终点才是成功！

### 逃生

很有趣的一道题。这里要求是编号小尽量靠前的排在前面，而不是最终队列字典序最小。比如说要求 3 在 1 前，那么队列应该是 3 1 2， 而不是 2 3 1。所以需要反向建图再拓扑排序。

### 确定比赛名次

和上一题相反。

### Rank of Tetris

首先合并已经明确排名相等的人，建图后拓扑排序。

1. 信息不完全：队列中多于一个人。
2. 信息冲突：图不是 DAG
   需要特别注意一下输出要求。

### Stock Chase

用 bitset 维护传递闭包。

### Legal or Not

floyd 判环。

### Reward

拓扑排序。

## 树链剖分，线段树

### CD操作

树上距离：$dis(a,b)=dep(a)+dep(b)-dep(lca)\times 2$

### How far away ？

类似上一题。

### Connections between cities

类似上一题。

### Gold Mine

不知道为什么把这道题放在这个专题。
这是我第一次接触到这个算法[最大权闭合图](https://www.cnblogs.com/wuyiqi/archive/2012/03/12/2391960.html)
对这道题的建图就是把价值连源点，花费连汇点，再把依赖关系连容量正无穷的边。

### Tree

区间修改，单点查询。

那么可以用差分的方式处理树链加减。  

### Dylans loves tree
因为 For each ② question,it guarantees that there is at most one value that appears odd times on the path.

那么路径上所有点异或和就是所求。

### The LCIS on the Tree
我的第一想法就是 dp 处理每条链的信息，但是合并时候太过繁琐，因为合并 LCA 所在的链需要调转方向，而且维护不完整的链也很乏力。  
但是我们可以用线段树来维护链的信息，每个节点记录含左右端点的和可以不含的最长上升连续序列，和下降序列的长度，并且记录左右端点的值。只需要重载 + 号，考虑怎么合并就行了。  
线段树的合并是显然的，合并路径上的链要复杂一些。我们发现其实和线段树的节点合并本质上是一样的。只是在合并 LCA ~ u 和 LCA ~ v 时候，我们发现需要把其中一条链翻转然后合并信息。也就是交换左右方向的值，把上升连续序列和下降连续序列的信息交换。
```cpp
struct node {
    int len, lc, fc, rc, LC, FC, RC, lv, rv;
    void init(int x = 0) {
        lc = fc = rc = LC = RC = FC = len = 1;
        lv = rv = x;
    }
    void reverse() {
        swap(rc, LC);
        swap(lc, RC);
        swap(FC, fc);
        swap(lv, rv);
    }
    node operator + (const node &x) const {
        if (x.lc == 0) return *this;
        if (lc == 0) return x;
        node b;
        b.len = len + x.len;
        b.lv = lv;
        b.rv = x.rv;
        b.lc = (lc == len) && (x.lv > rv) ? lc + x.lc : lc;
        b.rc = (x.rc == x.len) && (x.lv > rv) ? rc + x.rc : x.rc;
        b.fc = max({fc, x.fc, (x.lv > rv ? x.lc + rc : 0)});
        b.LC = (LC == len) && (x.lv < rv) ? LC + x.LC : LC;
        b.RC = (x.RC == x.len) && (x.lv < rv) ? RC + x.RC : x.RC;
        b.FC = max({FC, x.FC, (x.lv < rv ? x.LC + RC : 0)});
        return b;
    }
};
```
### Query on a tree 
我们只用离线询问，按 x 递减的顺序处理就行。  

### Relief grain
很有意思的一道题。  
我们单独考虑，如果只是一条链（也就是一个序列），我们怎么做。  
如果把所有操作汇成一张表格：
![](https://cdn.jsdelivr.net/gh/ricofx2003/Pictures@latest/Pic/H.png)
维护一个线段树，保存每种粮食的数量。逐列处理信息，那么到每个人时，线段树维护的就是他的粮食情况。线段树查询时在最大值的前提下尽量走左子树就可以保证编号最小。

### Occupation 
简单模拟。

### Tree chain problem 
考虑树上 dp。dp[v] 表示 v 为根节点的子树选择包含的链的最大价值。合并时，考虑加入 u 节点后新增的链，也就是以 u 为 LCA 的链。
![](https://cdn.jsdelivr.net/gh/ricofx2003/Pictures//Pic/K.png)

### Little Devil I 
>There is an old country and the king fell in love with a devil. The devil always asks the king to do some crazy things. Although the king used to be wise and beloved by his people. Now he is just like a boy in love and can’t refuse any request from the devil. Also, this devil is looking like a very cute Loli.

我看这个题目描述很行。  
以前看到这种题肯定无从下手，但这次我很快就有思路了，是不是有进步了呢？
![](https://cdn.jsdelivr.net/gh/ricofx2003/Pictures//Pic/Little%20Devil%20I.png)
用每个节点 u 代表 $fa_u$ 到 $u$ 这条边。
操作 1 很显然，维护一颗支持异或操作的线段树。  
对于操作 2，单独开一颗线段树，支持的操作和 1 相同。节点的值改变代表它所在的子树颜色改变(不包含它自己)。所以操作 2 也可以简单维护。  
最后我们发现，每个点的状态其实只受自己和父亲节点影响。
合并的时候套路和“The LCIS on the Tree”是一致的。

### The Water Problem 
很符合名字。

### Interviewe
这题有点问题，我也不会做。
参考: [https://vjudge.net/problem/HDU-3486](https://vjudge.net/problem/HDU-3486)

### GCD
1. 求区间 GCD 值，用线段树或者 st 表都可以维护。
2. 求 GCD 等于某个值的区间数量

问题 2 是一个经典问题。
搬一下我以前写的题解。

题目： [UVa1642](https://www.luogu.com.cn/problem/UVA1642)

因为要求一个最优的子序列，可以想到枚举这个子序列的右端点$j$。  那么怎么快速算出左端点$i$的答案呢？  
枚举每一个左端点，如果能知道这个子序列所有元素的$gcd$值就好了。  
先考虑这样一个序列$5,8,8,6,2$，假设现在$j=4$,可以算出所有子序列对应的$gcd$。
1. $i=1$, $gcd(a_1,a_2,a_3,a_4)=1$
2. $i=2$, $gcd(a_2,a_3,a_4)=2$
3. $i=3$, $gcd(a_3,a_4)=2$
4. $i=4$, $gcd(a_4)=6$

注意到不同子序列的$gcd$值有可能是相等的，事实上$gcd$值的种类最多不会超过$\log_2 a_j$个，因为$a_j$的约数个数一定不多于$\log_2 a_j$。  
上表从下向上看，每次增加一个元素的时候，$gcd$值是不变或者减小的，而且变小时一定会变成$a_j$的一个约数。所以就维护每个左端点对应的区间$gcd$值(即$gcd(a[i],a[i+1],...,a[j])$)，增加一个元素(即$j$ -> $j+1$)时，更新加上该元素后的区间$gcd$值就可以。  
知道了当前每个左端点对应的$gcd$，区间长度也是已知的，就可以计算答案了，对于每个区间的结果取一个最大值。  
但是直接这样做复杂度是不对的，$O(n^2log n)$显然过不了。那怎么办呢。
我们可以发现，如果两个左端点的$gcd$相等，那么$i$更小的一定会更优，所以直接把劣的删除，对后面不造成影响。这样一来每次要枚举的$i$就和$gcd$的个数有关，复杂度是$O(nlog^2 n)$，可以通过本题。


```cpp
#include<bits/stdc++.h>
using namespace std;

typedef long long ll;
const int N = 1e5 + 5;
ll n, a[N];
vector<pair<int, ll> > v;

ll read() {
	ll w = 0; char ch = getchar();
	while(ch > '9' || ch < '0') ch = getchar();
	while(ch >= '0' && ch <= '9') {
		w = w * 10 + ch - 48;
		ch = getchar();
	}
	return w;
}
ll gcd(ll a, ll b) {
	return b == 0 ? a : gcd(b, a % b);
}

int main() {
    ll T = read();
    while(T--) {
    	ll ans = 0; v.clear();
    	n = read();
    	for(int i = 1; i <= n; i++) a[i] = read();
        for(int j = 1; j <= n; j++) {
            v.push_back({j, a[j]});
            for(int i = v.size() - 2; i >=0; i--) {
            	v[i].second = gcd(v[i].second, a[j]);
            	if(v[i].second == v[i + 1].second) v.erase(v.begin() + i + 1);
            }
            for(int i = 0; i < v.size(); i++) {
            	ans = max(ans, (j - v[i].first + 1) * (v[i].second));
            }
        }
        printf("%lld\n", ans);
    }
    return 0;
}
```

### Taotao Picks Apples
这是一道好题。  
用线段树维护区间最大值，按题目要求能选出的数个数, 记作 cnt。
怎么合并某两个区间？假设左边区间最大值是 X, 那么我们需要知道右区间只选大于 X 的数，能选出多少个（不要求第一个必须选）。
比如查询线段树上某个节点 x ： query(l, r, lim)
我们可以分类讨论: 
1. MAX[lson] <= lim, query(l, r, lim) = query(mid + 1, r, lim)
2. MAX[lson] > lim, query(l, r, lim) = cnt[x] - cnt[lson] + query(l, mid, lim)

### Glad You Came 
吉老师线段树模板题。

## 区间 dp
### Dire Wolf 
状态设计：dp[i][j] 表示不猎取 i, j，但其他都被猎杀时，受到的最小攻击。

### You Are the One
改变顺序的方式相当于是一个栈，最先进去的会最后出来。
设 dp[i][j] 表示区间 (i,j) 的最小花费。转移时枚举第一个人放到哪里。注意这里每个区间都从 1 开始标号。

### Palindrome subsequence
设 dp[i][j] 表示 (i,j) 回文子序列数量。
```cpp
dp[i][j] = 0;
if (s[i] == s[j]) {
    dp[i][j] = (dp[i + 1][j] + dp[i][j - 1] + 1) % mod;
} else {
    dp[i][j] = (dp[i + 1][j] + dp[i][j - 1] + mod - dp[i + 1][j - 1]) % mod;
}
```

### Two Rabbits
可以发现两只兔子走过的路总长度不会大于 n。
只用求所有长度为 n 的区间的最长回文串长度就行。
另外还要考虑两只兔子从同一个起点出发。

### QSC and Master 
设 dp[i][j] 表示 (i,j) 的最大收益。
当计算 dp[i][j] 时：
1. 可以从 dp[i][k] + dp[k + 1][j] 转移而来，意思是 (k,k+1) 没有在一起被删除
2. 如果 (i + 1，j - 1) 可以被全部删除并且 i 和 j 可以作为一对删除，那么也可以从 dp[i + 1][j - 1] + a[i] + a[j] 转移

### String painter 
和上一题差不多的思路。
但是要考虑情况 2 时，会影响到中间的颜色。于是可以先预处理某个区间全是某个颜色时的最小步数。

### Jam's maze
想象有两个人，一个从(1,1)出发，另一个从(n,n)出发。每次走相同的字符的格子，最后在副对角线上回合，有多少种走法。
设 dp[step][x1][y1][x2][y2] 表示方案数。
这个状态可以精简，因为可以用横坐标计算出纵坐标。
并且第一维也可以用滚动数组优化。

### Expression
设 dp[i][j] 表示答案。
合并两个子区间时候，要考虑增加的操作方式。比如说两边都得到和之前相同的结果，但是有$\binom{j-i}{k-i}种新得操作方式，也就是左右自区间分别按原顺序可以交替操作。
1. 乘法：$(a_1+a_2+...+a_k)\times (b_1+b_2+...b_k)$
2. 加法： 加分和乘法不同，不能简单直接相加。考虑左边得到某个结果 x， 这个 x 实际上最后出现了 (右区间操作符号个数)！次

### Fragrant numbers 
如果能够表达出某个数，需要的区间长度不会很长。  
~~好吧我也不会证明。~~
于是暴力求出每个区间能够表达出哪些数字。

未完待续、、、、、、