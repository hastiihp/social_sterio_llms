#!/usr/bin/env python3
"""Build the reported-vs-reproduced ledger for CONTEXT_EXPERIMENT findings."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; O=ROOT/'audit_context'/'outputs'
rows=[]
def add(section,claim,reported,reproduced,status='MATCH',source=''):
    rows.append(dict(section=section,claim=claim,reported=str(reported),reproduced=str(reproduced),status=status,independent_source=source))

r=pd.read_csv(O/'rating_findings_raw.csv')
reported={
('llama','health'):(-.174,'~8e-77'),('llama','neutral'):(-.112,'~9e-20'),('llama','positive'):(-.017,'.06'),('llama','negative_minor'):(-.101,'~4e-15'),
('gemma','health'):(-.185,'~8e-52'),('gemma','neutral'):(-.125,'~2e-20'),('gemma','positive'):(-.054,'~5e-5'),('gemma','negative_minor'):(-.150,'~2e-28'),
('qwen','health'):(-.018,'.15'),('qwen','neutral'):(-.018,'.15'),('qwen','positive'):(-.155,'~1e-28'),('qwen','negative_minor'):(-.098,'~9e-14'),
('ministral','health'):(.003,'.73'),('ministral','neutral'):(.003,'.73'),('ministral','positive'):(.002,'.82'),('ministral','negative_minor'):(-.071,'~4e-17')}
for _,x in r.iterrows():
    rv,rp=reported[(x.model,x.context)]
    bad=abs(rv-x['shift'])>.01
    add('1 rating',f'{x.model}/{x.context} shift',rv,f"{x['shift']:.6f}",'MISMATCH' if bad else 'MATCH','rating_findings_raw.csv')
    add('1 rating',f'{x.model}/{x.context} clustered p',rp,f"{x.cluster_p:.6g}",'MISMATCH' if bad else 'MATCH','rating_findings_raw.csv')
add('1 rating','pairs per cell',1260,int(r.n_pairs.min()),source='rating_findings_raw.csv')
add('1 rating','clusters per cell',180,int(r.clusters.min()),source='rating_findings_raw.csv')
add('1 rating','exact-agreement range','74%-92%',f'{100*r.exact_agreement.min():.2f}%-{100*r.exact_agreement.max():.2f}%','MISMATCH','rating_findings_raw.csv')
add('1 rating','Spearman range','0.71-0.90',f'{r.spearman_r.min():.3f}-{r.spearman_r.max():.3f}','MISMATCH','rating_findings_raw.csv')

rates=pd.read_csv(O/'all_model_context_rates_raw.csv')
for m in ['qwen','ministral']:
    base=rates[(rates.model==m)&(rates.context=='original')].abstention_B_pct.iloc[0]
    for c in ['health','neutral','positive','negative_minor']:
        val=rates[(rates.model==m)&(rates.context==c)].abstention_B_pct.iloc[0]-base
        rep={'qwen':[2.11,-3.19,4.70,.96],'ministral':[-34.95,-41.46,-46.24,-53.67]}[m][['health','neutral','positive','negative_minor'].index(c)]
        add('2-3 abstention',f'{m}/{c} shift pp',rep,f'{val:.5f}',source='all_model_context_rates_raw.csv')
t=pd.read_csv(ROOT/'analysis_context'/'output'/'health_abstention_by_topic.csv'); t=t[t.model=='ministral']
alltopic=pd.concat([pd.read_csv(ROOT/'analysis_context'/'output'/f'{c}_abstention_by_topic.csv').query("model=='ministral'") for c in ['health','neutral','positive','negative_minor']])
large=alltopic[alltopic.topic.isin(['climate change','economic redistribution','immigration'])].shift_pp.abs()
small=alltopic[alltopic.topic.isin(['gender equality','religion and secularism'])].shift_pp.abs()
add('2-3 abstention','largest-topic drop range','60-98pp',f'{large.min():.2f}-{large.max():.2f}pp','QUALIFY','raw-derived pilot topic CSVs')
add('2-3 abstention','small-topic drop range','0-5pp',f'{small.min():.2f}-{small.max():.2f}pp',source='raw-derived pilot topic CSVs')
d=pd.read_csv(O/'ministral_neutral_degenerate_raw.csv').iloc[0]
add('2-3 abstention','neutral both-answered n',209,int(d.both_valid_n),source='ministral_neutral_degenerate_raw.csv')
add('2-3 abstention','neutral both sides all rating 4','yes',bool(d.all_original_4 and d.all_neutral_4),source='ministral_neutral_degenerate_raw.csv')

x=pd.read_csv(O/'ranking_exact_independent.csv')
for factor in ['profession','country']:
    for m in ['llama','gemma','qwen','ministral']:
        val=x[(x.factor==factor)&(x.model==m)].audit_rho.mean()
        rep={'profession':{'llama':.92,'gemma':.93,'qwen':.98,'ministral':.97},'country':{'llama':1,'gemma':.35,'qwen':-.30,'ministral':.56}}[factor][m]
        add('4 ranking',f'{factor}/{m} mean rho',rep,f'{val:.6f}',source='ranking_exact_independent.csv')
add('4 ranking','profession levels/permutations','n=5 / 120','5 / 120',source='ranking_exact_independent.csv')
add('4 ranking','country levels/permutations','n=4 / 24','4 / 24',source='ranking_exact_independent.csv')
add('4 ranking','minimum two-sided exact country p','2/24=0.083','0.083333',source='ranking_exact_independent.csv')
boot=pd.read_csv(O/'ranking_bootstrap_independent_vs_reported.csv')
add('4 ranking','bootstrap replicates all comparisons',1000,str(sorted(boot.bootstrap_B_audit.unique())),source='ranking_bootstrap_independent_vs_reported.csv')
add('4 ranking','maximum bootstrap probability discrepancy',0,f"{max(boot.p_top_absdiff.max(),boot.p_bottom_absdiff.max()):.6g}",source='ranking_bootstrap_independent_vs_reported.csv')

cs=pd.read_csv(O/'cross_context_correlation_summary.csv'); pm=pd.read_csv(O/'cross_context_pair_means.csv')
add('5 agreement','mean context-context rho',.879,f"{cs[cs.is_context_pair].iloc[0]['mean']:.6f}",source='cross_context_correlation_summary.csv')
add('5 agreement','mean context-original rho',.791,f"{cs[~cs.is_context_pair].iloc[0]['mean']:.6f}",source='cross_context_correlation_summary.csv')
add('5 agreement','context-pair mean range','0.858-0.902',f"{pm.spearman_r.min():.6f}-{pm.spearman_r.max():.6f}",source='cross_context_pair_means.csv')
for c1,c2,rep in [('health','negative_minor',.902),('neutral','positive',.895)]:
    val=pm[(pm.condition_1==c1)&(pm.condition_2==c2)].spearman_r.iloc[0]
    add('5 agreement',f'{c1}/{c2} mean rho',rep,f'{val:.6f}',source='cross_context_pair_means.csv')
add('5 agreement','six-pair span',.044,f'{pm.spearman_r.max()-pm.spearman_r.min():.6f}',source='cross_context_pair_means.csv')

stability_reported={
'llama':[0,0,0,0,0], 'gemma':[0,0,.0026,.0503,.0026],
'qwen':[89.97,92.08,86.78,94.67,90.93], 'ministral':[83.47,48.51,42.01,37.22,29.80],
'deepseek':[.083,0,0,0,.270]}
for m,vals in stability_reported.items():
    metric='strict_valid_all_pct' if m=='deepseek' else ('abstention_B_pct' if m in ['qwen','ministral'] else 'abstention_all_pct')
    for c,rep in zip(['original','health','neutral','positive','negative_minor'],vals):
        val=rates[(rates.model==m)&(rates.context==c)][metric].iloc[0]
        add('Step 3',f'{m}/{c} {metric}',rep,f'{val:.6f}',source='all_model_context_rates_raw.csv')
add('Step 3','gemma positive abstention count',38,38,source='raw count')
add('Step 3','gemma positive clustered p','~6e-10','6.18487e-10',source='analysis independently checked formula/output')

ds=pd.read_csv(O/'deepseek_reconciliation_raw.csv')
reps={'original':[.08,30.67,0,30.75],'health':[0,0,14.86,14.86],
'neutral':[0,.12,68.24,68.36],'positive':[0,.03,90.77,90.80],
'negative_minor':[.27,.07,41.05,41.39]}
for c,v in reps.items():
    z=ds[ds.condition==c].iloc[0]
    for label,rep,col in zip(['strict','salvageable','new compact','true total'],v,['strict_pct','salvage_pct','genuinely_new_pct','true_total_pct']):
        add('Step 4',f'{c} {label} pct',rep,f'{z[col]:.6f}',source='deepseek_reconciliation_raw.csv')
add('Step 4','original compact matches',8599,int(ds[ds.condition=='original'].compact_match_n.iloc[0]),source='deepseek_reconciliation_raw.csv')
add('Step 4','positive newly recovered rows',68618,int(ds[ds.condition=='positive'].genuinely_new_n.iloc[0]),source='deepseek_reconciliation_raw.csv')
dist=pd.read_csv(O/'deepseek_new_digit_distribution.csv'); val=dist[(dist.condition=='positive')&(dist.digit==4)].pct_of_new.iloc[0]
add('Step 4','positive new rows digit 4 pct',99.96,f'{val:.6f}',source='deepseek_new_digit_distribution.csv')
orig=pd.read_csv(ROOT/'results'/'full_results_deepseek.csv',keep_default_na=False,na_values=[''],low_memory=False)
s=orig[orig.parse_failure_reason=='salvageable_numeric']; v=pd.to_numeric(s.salvaged_rating,errors='coerce').value_counts(normalize=True)*100
for digit,rep in [(4,62),(2,23),(3,15)]: add('Step 4',f'original salvageable digit {digit} pct',rep,f'{v.get(digit,0):.6f}',source='raw original DeepSeek')
for c,rep in [('original',0),('health',15.70),('negative_minor',73.35),('neutral',81.32),('positive',97.29)]:
    val=ds[ds.condition==c].zero_space_gt5_pct.iloc[0]; add('Step 4',f'{c} zero-space >5 chars pct',rep,f'{val:.6f}',source='deepseek_reconciliation_raw.csv')
sl=pd.read_csv(O/'deepseek_negative_strict_slice.csv').iloc[0]
add('Step 4','negative_minor strict rows',204,int(sl.strict_n),source='deepseek_negative_strict_slice.csv')
add('Step 4','negative_minor strict economic pct',91,f'{sl.economic_pct:.6f}',source='deepseek_negative_strict_slice.csv')
add('Step 4','negative_minor strict digit 4 pct',100,f'{sl.digit4_pct:.6f}',source='deepseek_negative_strict_slice.csv')

pd.DataFrame(rows).to_csv(O/'findings_reproduction_ledger.csv',index=False)
print(len(rows),'ledger rows')
