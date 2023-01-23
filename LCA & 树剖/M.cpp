#include <bits/stdc++.h>
using namespace std;

int n, st[11][1005];

int main() {
    int T;
    scanf("%d", &T);
    while  (T--) {
        scanf("%d", &n);
        for (int i = 1; i <= n; i++) scanf("%d", &st[0][i]);
        for (int j = 1; j <= 10; j++) {
            for (int i = 1; i <= n; i++) {
                st[j][i] = max(st[j - 1][i], st[j - 1][i + (1 << (j - 1))]);
            }
        }
        int q;
        scanf("%d", &q);
        while (q--) {
            int l, r;
            scanf("%d %d", &l, &r);

        }
    }
    return 0;
}