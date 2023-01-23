#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mod = 998244353;
const int I[11] = {0, 1, 1, 4, 5, 1, 4, 1, 9, 1, 9};
int a[105], ans[5005];
vector<int> f[21][21];

int sub(int l, int r) {
    if (r - l + 1 > 4) return -1;
    int re = 0;
    for (int i = l; i <= r; i++) re = re * 10 + a[i];
    return re;
}

int main() {
    for (int i = 1; i < 104; i++) a[i] = I[(i - 1) % 10 + 1];
    for (int len = 1; len <= 10; len++) {
        for (int i = 1; i + len - 1 <= 10; i++) {
            int j = i + len - 1;
            if (sub(i, j) != -1) f[i][j].push_back(sub(i, j));
            for (int k = i; k < j; k++) {
                for (int x : f[i][k]) {
                    for (int y : f[k + 1][j]) {
                        if (x * y <= 5000) f[i][j].push_back(x * y);
                        if (x + y <= 5000) f[i][j].push_back(x + y);
                    }
                }
            }
        }
    }
    memset(ans, -1, sizeof ans);
    for (int j = 1; j <= 15; j++) for (int x : f[1][j]) {
        if (ans[x] == -1) ans[x] = j;
    }
    int num;
    scanf("%d", &num);
    while (num--) {
        int x;
        scanf("%d", &x);
        printf("%d\n", ans[x]);
    }
    return 0;
}