#include <bits/stdc++.h>
using namespace std;

const int N = 100005;
int gc[N << 2], n, a[N], gg[N];
unordered_map<int, long long> mp;
vector<pair<int, int>> v;
#define lson (x << 1)
#define rson ((x << 1) | 1)
#define  mid ((l + r) >> 1)
int gcd(int x, int y) {
    return y ? gcd(y, x % y) : x;
}
void build(int x, int l, int r) {
    if (l == r) {
        gc[x] = a[l];
        return;
    }
    build(lson, l, mid);
    build(rson, mid + 1, r);
    gc[x] = gcd(gc[lson], gc[rson]);
}
int query(int x, int l, int r, int L, int R) {
    if (l > R || r < L) return 0;
    if (l >= L && r <= R) return gc[x];
    return gcd(query(lson, l, mid, L, R), query(rson, mid + 1, r, L, R)); 
}

void rmain() {
    mp.clear();
    v.clear();
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) scanf("%d", &a[i]);
    build(1, 1, n);
    int q;
    scanf("%d", &q);
    for (int i = 1; i <= q; i++) {
        int l, r;
        scanf("%d %d", &l, &r);
        gg[i] = query(1, 1, n, l, r);
    }
    for (int i = 1; i <= n; i++) {
        v.push_back({i, a[i]});
        for (int j = (int)(v.size()) - 2; j >= 0; j--) {
            v[j].second = gcd(v[j].second, a[i]);
            if (v[j].second == v[j + 1].second) v.erase(v.begin() + j + 1);
        }
        mp[v[(int)(v.size()) - 1].second] += i - v[(int)(v.size()) - 1].first + 1;
        for (int j = (int)(v.size()) - 2; j >= 0; j--) {
            mp[v[j].second] += v[j + 1].first - v[j].first;
        }
    }
    for (int i = 1; i <= q; i++) printf("%d %lld\n", gg[i], mp[gg[i]]);
}

int main() {
    int T, C = 0;
    scanf("%d", &T);
    while (T--) {
        printf("Case #%d:\n", ++C);
        rmain();
    }
    return 0;
}