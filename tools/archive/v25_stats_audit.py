"""Statistical cross-check of v2.5 trajectory claims."""
import json
import numpy as np
from math import sqrt
from scipy import stats

ep_log = json.load(open('C:/Users/jarms/repos/clark/clark/data/logs/pretrain/episode_log.json'))
eps = ep_log['episodes']
v25 = eps[-252:]
prior = eps[:-252]

FILLER = {'side_project','loading','training','quality_check','returns_processing','receiving'}
def eppct(e):
    wt = e['footer'].get('worker_time', {}) or {}
    agg = {}
    for w_name, tasks in wt.items():
        for t, h in tasks.items():
            agg[t] = agg.get(t, 0) + h
    total = sum(agg.values()) or 1
    return 100 * sum(h for t,h in agg.items() if t in FILLER) / total

def is_heavy(e):
    n_w = len(e['footer'].get('worker_time', {}))
    return n_w >= 20 or e['header'].get('total_orders', 0) >= 400

# --- v2.5 vs prior baseline comparison ---
print("=" * 60)
print("v2.5 (last 252) vs PRIOR (270 eps, v2.4-projection + leftovers)")
print("=" * 60)

# Rewards
r_v25 = [e['footer'].get('reward', 0) for e in v25]
r_prior = [e['footer'].get('reward', 0) for e in prior]
t, p = stats.ttest_ind(r_v25, r_prior, equal_var=False)
print(f"\nMean reward: prior {np.mean(r_prior):.0f}, v2.5 {np.mean(r_v25):.0f}, "
      f"delta {np.mean(r_v25)-np.mean(r_prior):+.0f}, Welch p={p:.5f}")

# Heavy filler
heavy_v25 = [eppct(e) for e in v25 if is_heavy(e)]
heavy_prior = [eppct(e) for e in prior if is_heavy(e)]
t, p = stats.ttest_ind(heavy_v25, heavy_prior, equal_var=False)
print(f"Heavy filler %: prior {np.mean(heavy_prior):.1f}%, v2.5 {np.mean(heavy_v25):.1f}%, "
      f"delta {np.mean(heavy_v25)-np.mean(heavy_prior):+.1f}pp, Welch p={p:.5f}")

# Heavy drown rate (>40% filler)
drown_v25 = sum(1 for f in heavy_v25 if f > 40)
drown_prior = sum(1 for f in heavy_prior if f > 40)
n1, n2 = len(heavy_v25), len(heavy_prior)
pp = (drown_v25 + drown_prior) / (n1 + n2)
se = sqrt(pp*(1-pp)*(1/n1 + 1/n2))
z = (drown_v25/n1 - drown_prior/n2) / max(1e-9, se)
p = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"Heavy drown (>40%): prior {100*drown_prior/n2:.1f}%, v2.5 {100*drown_v25/n1:.1f}%, "
      f"delta {100*(drown_v25/n1-drown_prior/n2):+.1f}pp, z={z:.2f} p={p:.4f}")

# Heavy ship rate
ship_v25 = [e['footer'].get('orders_shipped',0)/max(1,e['footer'].get('orders_total',1))
            for e in v25 if is_heavy(e)]
ship_prior = [e['footer'].get('orders_shipped',0)/max(1,e['footer'].get('orders_total',1))
              for e in prior if is_heavy(e)]
t, p = stats.ttest_ind(ship_v25, ship_prior, equal_var=False)
print(f"Heavy ship %: prior {100*np.mean(ship_prior):.1f}%, v2.5 {100*np.mean(ship_v25):.1f}%, "
      f"delta {100*(np.mean(ship_v25)-np.mean(ship_prior)):+.1f}pp, Welch p={p:.5f}")

# --- Within-v2.5 trajectory check ---
print("\n" + "=" * 60)
print("WITHIN v2.5: ep 1-140 vs ep 141-252")
print("=" * 60)
v25_first = v25[:140]
v25_later = v25[140:]

# A-rate
a_f = sum(1 for e in v25_first if e['footer'].get('grade')=='A')
a_l = sum(1 for e in v25_later if e['footer'].get('grade')=='A')
pp_a = (a_f+a_l)/252
se_a = sqrt(pp_a*(1-pp_a)*(1/140+1/112))
z_a = (a_f/140 - a_l/112)/max(1e-9,se_a)
p_a = 2*(1-stats.norm.cdf(abs(z_a)))
print(f"A-rate: ep1-140 {100*a_f/140:.1f}%, ep141-252 {100*a_l/112:.1f}%, "
      f"z={z_a:.2f} p={p_a:.4f}")

# F-rate
f_f = sum(1 for e in v25_first if e['footer'].get('grade')=='F')
f_l = sum(1 for e in v25_later if e['footer'].get('grade')=='F')
pp_f = (f_f+f_l)/252
se_f = sqrt(pp_f*(1-pp_f)*(1/140+1/112))
z_f = (f_f/140 - f_l/112)/max(1e-9,se_f)
p_f = 2*(1-stats.norm.cdf(abs(z_f)))
print(f"F-rate: ep1-140 {100*f_f/140:.1f}%, ep141-252 {100*f_l/112:.1f}%, "
      f"z={z_f:.2f} p={p_f:.4f}")

# Heavy drown within v2.5
hv_first = [eppct(e) for e in v25_first if is_heavy(e)]
hv_later = [eppct(e) for e in v25_later if is_heavy(e)]
drown_f = sum(1 for f in hv_first if f > 40)
drown_l = sum(1 for f in hv_later if f > 40)
pp_d = (drown_f+drown_l)/(len(hv_first)+len(hv_later))
se_d = sqrt(pp_d*(1-pp_d)*(1/len(hv_first)+1/len(hv_later)))
z_d = (drown_f/len(hv_first) - drown_l/len(hv_later))/max(1e-9,se_d)
p_d = 2*(1-stats.norm.cdf(abs(z_d)))
print(f"Heavy drown: ep1-140 {100*drown_f/len(hv_first):.1f}%, "
      f"ep141-252 {100*drown_l/len(hv_later):.1f}%, z={z_d:.2f} p={p_d:.4f}")

# 50-ep chunk A-rate
print("\n50-ep chunk A-rate trajectory:")
for i in range(0, 252, 50):
    block = v25[i:i+50]
    if not block: continue
    a = sum(1 for e in block if e['footer'].get('grade')=='A')
    print(f"  ep {i+1:>3}-{i+len(block):>3}: {a}/{len(block)} A ({100*a/len(block):.0f}%)")
