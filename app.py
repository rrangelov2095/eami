import os

from pathlib import Path
import math
import base64
import html
import io
import re
from datetime import datetime
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, KeepTogether
from reportlab.lib.utils import ImageReader

APP_DIR = Path(__file__).resolve().parent
REPORT_DIR = APP_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)
df = pd.read_csv(APP_DIR / "eami_reference_v03.csv")

IMMUNE = [
    "FAS","NFKB1","CASP1","NLRP3","IFIH1","TLR5","ADAR","MDM2",
    "STING1","TLR3","TLR4","BBC3","TP53","IL18","OAS1"
]
SCMC = [
    "NLRP7","KHDC3L","OOEP","PADI6","PEG3","DDX39B","PRPF8",
    "PARP1","BRCA1","RAD51"
]
BRCA = [
    "BRCA1_ENST00000354071_8_FPKM",
    "BRCA1_ENST00000618469_2_FPKM",
    "BRCA1_ENST00000634433_2_FPKM",
]
MT_VARIANTS = [
    "mtvar_2705_MT_RNR2",
    "mtvar_3009_MT_RNR2",
    "mtvar_8696_MT_ATP6",
    "mtvar_14765_MT_CYB",
    "mtvar_14904_MT_CYB",
    "mtvar_15451_MT_CYB",
    "mtvar_15939_noncoding",
]
MT_VARIANT_LABELS = [
    "2705 · MT-RNR2",
    "3009 · MT-RNR2",
    "8696 · MT-ATP6",
    "14765 · MT-CYB",
    "14904 · MT-CYB",
    "15451 · MT-CYB",
    "15939 · non-coding",
]

ISOFORM_COLS = ['iso_01', 'iso_02', 'iso_03', 'iso_04', 'iso_05', 'iso_06', 'iso_07', 'iso_08', 'iso_09', 'iso_10', 'iso_11', 'iso_12', 'iso_13', 'iso_14', 'iso_15', 'iso_16', 'iso_17', 'iso_18', 'iso_19', 'iso_20', 'iso_21', 'iso_22', 'iso_23', 'iso_24', 'iso_25']
ISOFORM_LABELS = ['ENST00000439477.1 · MTND6P18', 'ENST00000481618.2 · NMTRS-TGA3-1', 'ENST00000373866.8 · BICC1', 'ENST00000379160.3 · PCNA', 'ENST00000860796.1 · HECTD1', 'ENST00000564733.1 · METTL9', 'ENST00000888879.1 · PROM1', 'ENST00000243326.9 · RIF1', 'ENST00000688377.1 · LINC02249', 'ENST00000514997.1 · MTND4P8', 'ENST00000430812.2 · MTND5P23', 'ENST00000528711.1 · MTND6P25', 'ENST00000453886.1 · MTND4P21', 'ENST00000402078.1 · MTND4P13', 'ENST00000936301.1 · HSPA4', 'ENST00000503354.1 · ECM1P2', 'ENST00000528255.1 · MTND2P26', 'ENST00000455055.1 · MTND2P8', 'ENST00000436242.1 · MTND2P21', 'ENST00000504006.1 · MTND6P16', 'ENST00000639031.1 · TAF11L13', 'ENST00000917949.1 · EIF2AK4', 'ENST00000298910.12 · LRRK2', 'ENST00000714332.1 · ENSG00000293571', 'ENST00000282516.13 · NIPBL']
ISOFORM_MTND_IDX = [0,9,10,11,12,13,16,17,18,19]
ISOFORM_MATERNAL_IDX = [20,21,22,23,24]

def _num(x, default=0.0):
    try:
        x = float(x)
        return default if math.isnan(x) else x
    except:
        return default

def percentile(series, value):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0 or value is None:
        return None
    try:
        v = float(value)
        if math.isnan(v):
            return None
    except:
        return None
    return float((s <= v).mean()*100.0)

def robust_z(series, value):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return 0.0
    v = _num(value)
    med = float(s.median())
    q1, q3 = float(s.quantile(.25)), float(s.quantile(.75))
    scale = q3-q1
    if scale <= 1e-12:
        scale = float(s.std()) or 1.0
    return (v-med)/scale


def technical_context(total_reads, detected_genes):
    """Cohort-position context only; not a validated QC pass/fail system."""
    rp = percentile(df["total_reads_approx"], total_reads)
    gp = percentile(df["detected_genes_approx"], detected_genes)

    if rp is None and gp is None:
        return {
            "status":"NOT ASSESSED",
            "read_percentile":None,
            "gene_percentile":None,
            "note":"Total reads and detected genes were not entered."
        }

    low_r = rp is not None and rp <= 25
    low_g = gp is not None and gp <= 25

    if low_r and low_g:
        status = "LOW COHORT POSITION"
        note = "Both sequencing depth and detected-gene count fall in the bottom quartile of the current 16-sample cohort."
    elif low_r or low_g:
        status = "CAUTION"
        note = "One technical-context metric falls in the bottom quartile of the current cohort."
    else:
        status = "WITHIN / ABOVE COHORT"
        note = "Entered technical-context metrics are not in the bottom quartile of the current cohort."

    return {
        "status":status,
        "read_percentile":rp,
        "gene_percentile":gp,
        "note":note
    }

def isoform_summary(vals, sample_type):
    entered = []
    user = []
    for c in ISOFORM_COLS:
        v = vals.get(c)
        if v is not None:
            try:
                fv = float(v)
                if not math.isnan(fv):
                    entered.append(c)
                    user.append(fv)
            except:
                pass

    if not entered:
        return {
            "available":False,
            "closest_id":None,
            "closest_sample":None,
            "similarity":None,
            "active_count":None,
            "mtnd_block":None,
            "maternal_block":None,
            "label":"Isoform panel not entered"
        }

    ref = df[df["sample_type"] == sample_type]
    ranked = []
    for _, r in ref.iterrows():
        diffs = []
        for c in entered:
            rv = r[c]
            if pd.isna(rv):
                continue
            diffs.append(abs(float(vals[c]) - float(rv)))
        if diffs:
            mae = float(np.mean(diffs))
            sim = max(0.0, 100.0*(1.0-mae))
            ranked.append((sim, r["biological_id"], r["sample_id"]))
    ranked.sort(reverse=True)

    def block_mean(indices):
        vv=[]
        for i in indices:
            c=ISOFORM_COLS[i]
            v=vals.get(c)
            if v is not None:
                try:
                    fv=float(v)
                    if not math.isnan(fv):
                        vv.append(fv)
                except:
                    pass
        return None if not vv else float(np.mean(vv))

    active = sum(float(vals[c]) >= 0.5 for c in entered)
    mtnd = block_mean(ISOFORM_MTND_IDX)
    maternal = block_mean(ISOFORM_MATERNAL_IDX)

    closest = ranked[0] if ranked else (None,None,None)
    label = "Isoform pattern available"
    if sample_type == "embryo" and closest[2] == "sample05" and closest[0] is not None and closest[0] >= 80:
        label = "P8_E3-like isoform pattern"
    elif sample_type == "embryo" and closest[2] == "sample17" and closest[0] is not None and closest[0] >= 80:
        label = "P7_E1-like isoform pattern"
    elif sample_type == "oocyte" and mtnd is not None and mtnd >= 0.6:
        label = "MTND-pseudogene-rich oocyte isoform pattern"

    return {
        "available":True,
        "closest_id":closest[1],
        "closest_sample":closest[2],
        "similarity":closest[0],
        "active_count":active,
        "mtnd_block":mtnd,
        "maternal_block":maternal,
        "label":label
    }


def _upper_tail_burden(percentile_value):
    """0 below/at median; rises linearly to 1 at the 100th percentile."""
    if percentile_value is None:
        return None
    return max(0.0, min(1.0, (float(percentile_value)-50.0)/50.0))

def _cohort_center_integrity(percentile_value):
    """100 at cohort median, 0 at either extreme. Used only for IR in ETIS beta candidate."""
    if percentile_value is None:
        return None
    return max(0.0, 100.0*(1.0-abs(float(percentile_value)-50.0)/50.0))

def etis_beta_candidate(vals, sample_type):
    """
    ETIS beta candidate v0.2
    ------------------------
    Cohort-relative research score with domain-specific biological weighting.

    Design changes relative to alpha:
    1) Immune/stress separates core damage programmes from sensor/adaptive programmes.
    2) Transcriptomic stability includes DDX39B–PEG3 corroboration.
    3) RNA-processing evidence can receive extra weight when extreme IR is corroborated by PARP1.
    4) Evidence-correlated domains can receive modest dynamic weight bonuses.
    5) No global bottleneck penalty is used.

    IMPORTANT:
    - Embryo mode only for the integrated beta candidate.
    - Technical context is excluded from ETIS.
    - chrM representation remains context-only.
    - Isoform switching remains visible but is not directly weighted in beta candidate v0.2.
    - The rules were developed after reviewing discordant cases in the same small cohort;
      agreement with provisional labels is NOT independent validation.
    """
    ref = df[df["sample_type"] == sample_type]

    # -------------------------
    # Mitochondrial domain
    # -------------------------
    vs = variant_summary(vals)
    mito_score = None
    mito_note = "Variant panel not available; chrM representation is context-only."
    if vs["available"]:
        mito_score = max(0.0, min(100.0, 100.0*(1.0-vs["burden_score"])))
        mito_note = "100 minus transcript-derived mt-variant burden; chrM itself is not directly scored."

    # -------------------------
    # Transcriptomic stability
    # -------------------------
    lp = percentile(ref["LINE1_RPM_approx"], vals.get("LINE1_RPM_approx"))
    cp = percentile(ref["chimeric_RPM_approx"], vals.get("chimeric_RPM_approx"))
    tail_burdens = [x for x in [_upper_tail_burden(lp), _upper_tail_burden(cp)] if x is not None]
    base_trans_burden = float(np.mean(tail_burdens)) if tail_burdens else None

    ddx_peg3 = gmean(vals.get("DDX39B"), vals.get("PEG3"))
    trans_burden = None
    transcript_score = None
    if base_trans_burden is not None:
        trans_burden = max(base_trans_burden, 0.85*ddx_peg3)
        transcript_score = max(0.0, min(100.0, 100.0*(1.0-trans_burden)))

    transcript_note = (
        "Upper-tail LINE-1/chimeric burden with DDX39B–PEG3 corroboration "
        f"(corroboration={ddx_peg3:.2f})."
    )

    # -------------------------
    # RNA-processing domain
    # -------------------------
    irp = percentile(ref["IR_index_approx"], vals.get("IR_index_approx"))
    rna_score = _cohort_center_integrity(irp) if sample_type == "embryo" else None
    parp1 = _num(vals.get("PARP1"))
    ir_extreme = bool(irp is not None and (irp >= 75.0 or irp <= 25.0))
    parp1_ir_corroboration = bool(parp1 >= 0.75 and ir_extreme)

    rna_note = (
        "IR cohort-center integrity score. "
        f"PARP1={parp1:.2f}; extreme-IR/PARP1 corroboration={'yes' if parp1_ir_corroboration else 'no'}. "
        "Isoform panel remains contextual in beta v0.2."
        if sample_type == "embryo"
        else "No integrated oocyte RNA-processing score in beta v0.2."
    )

    # -------------------------
    # Immune / stress domain
    # -------------------------
    core_programs = [
        gmean(vals.get("FAS"), vals.get("NFKB1")),
        gmean(vals.get("TP53"), vals.get("BBC3")),
        gmean(vals.get("NLRP3"), vals.get("CASP1")),
    ]
    sensor_programs = [
        gmean(vals.get("IFIH1"), vals.get("TLR5")),
        gmean(vals.get("STING1"), vals.get("MDM2")),
        _num(vals.get("TLR3")),
    ]
    core_damage = max(core_programs) if core_programs else 0.0
    sensor_activity = max(sensor_programs) if sensor_programs else 0.0

    # Sensor/adaptive activity is intentionally softer than corroborated
    # apoptotic/inflammasome damage programmes.
    SENSOR_PENALTY_FACTOR = 0.35
    immune_burden = max(core_damage, SENSOR_PENALTY_FACTOR*sensor_activity)
    immune_score = max(0.0, min(100.0, 100.0*(1.0-immune_burden)))
    immune_note = (
        f"Core damage={core_damage:.2f}; sensor/adaptive={sensor_activity:.2f}; "
        f"sensor penalty factor={SENSOR_PENALTY_FACTOR:.2f}."
    )

    # -------------------------
    # SCMC / DDR domain
    # -------------------------
    preservation_vals = [
        _num(vals.get("NLRP7")),
        _num(vals.get("KHDC3L")),
        _num(vals.get("PADI6")),
        _num(vals.get("BRCA1")),
    ]
    scmc_score = max(0.0, min(100.0, 100.0*float(np.mean(preservation_vals))))
    scmc_note = "Mean preservation signal of NLRP7, KHDC3L, PADI6 and BRCA1."

    # -------------------------
    # Candidate base weights
    # -------------------------
    base_weights = {
        "Mitochondrial integrity": 0.15,
        "Transcriptomic stability": 0.20,
        "RNA processing integrity": 0.25,
        "Immune / stress integrity": 0.15,
        "SCMC / DDR integrity": 0.25,
    }
    candidate_weights = dict(base_weights)

    # Evidence-specific weighting: modest bonuses only when an additional
    # independent/corroborating signal supports the domain abnormality.
    if ddx_peg3 >= 0.75:
        candidate_weights["Transcriptomic stability"] += 0.10
    if parp1_ir_corroboration:
        candidate_weights["RNA processing integrity"] += 0.15
    if core_damage >= 0.75:
        candidate_weights["Immune / stress integrity"] += 0.10

    domain_scores = {
        "Mitochondrial integrity": mito_score,
        "Transcriptomic stability": transcript_score,
        "RNA processing integrity": rna_score,
        "Immune / stress integrity": immune_score,
        "SCMC / DDR integrity": scmc_score,
    }
    domain_notes = {
        "Mitochondrial integrity": mito_note,
        "Transcriptomic stability": transcript_note,
        "RNA processing integrity": rna_note,
        "Immune / stress integrity": immune_note,
        "SCMC / DDR integrity": scmc_note,
    }

    available_names = [n for n,s in domain_scores.items() if s is not None]
    completeness = 100.0 * len(available_names) / 5.0

    normalized_weights = {}
    if available_names:
        denom = sum(candidate_weights[n] for n in available_names)
        if denom <= 0:
            denom = float(len(available_names))
            normalized_weights = {n:1.0/len(available_names) for n in available_names}
        else:
            normalized_weights = {n:candidate_weights[n]/denom for n in available_names}

    domain_rows = []
    for name in domain_scores:
        score = domain_scores[name]
        included = score is not None
        domain_rows.append({
            "domain": name,
            "score": score,
            "included": included,
            "base_weight": base_weights[name],
            "candidate_weight": candidate_weights[name],
            "effective_weight_pct": (100.0*normalized_weights[name]) if name in normalized_weights else None,
            "rule": domain_notes[name],
        })

    if sample_type != "embryo":
        etis = None
        band = "OOCYTE ETIS NOT YET DEFINED"
        band_note = "Oocyte domains remain descriptive; no integrated oocyte ETIS is claimed in beta v0.2."
    elif len(available_names) < 3:
        etis = None
        band = "INSUFFICIENT DOMAIN COVERAGE"
        band_note = "At least 3 of the 5 beta-candidate domains are required."
    else:
        etis = float(sum(normalized_weights[n]*domain_scores[n] for n in available_names))
        if etis >= 85.0:
            band = "LOW MOLECULAR DEVIATION · CLEAN-like β"
        elif etis >= 45.0:
            band = "INTERMEDIATE MOLECULAR DEVIATION"
        else:
            band = "HIGH MOLECULAR DEVIATION · DIRTY-like β"
        band_note = (
            "Candidate thresholds (45 / 85) retained from alpha for comparison. "
            "The beta rules were derived within the same small cohort and are not independently validated."
        )

    return {
        "score": etis,
        "band": band,
        "band_note": band_note,
        "completeness": completeness,
        "domains": domain_rows,
        "version": "ETIS beta candidate v0.2",
        "details": {
            "ddx39b_peg3": ddx_peg3,
            "parp1_ir_corroboration": parp1_ir_corroboration,
            "core_damage": core_damage,
            "sensor_activity": sensor_activity,
            "sensor_penalty_factor": SENSOR_PENALTY_FACTOR,
        },
    }

def variant_summary(vals):
    coding_fields = MT_VARIANTS[:-1]
    known = [vals.get(f) for f in MT_VARIANTS if vals.get(f) is not None]
    coding = [_num(vals.get(f)) for f in coding_fields if vals.get(f) is not None]
    noncoding = vals.get("mtvar_15939_noncoding")

    if not known:
        return {
            "available": False, "coding_count20": None, "total_count20": None,
            "max_fraction": None, "label": "Variant panel not entered",
            "burden_score": 0.0
        }

    allv = [_num(v) for v in known]
    coding_count = sum(v >= .20 for v in coding)
    total_count = sum(v >= .20 for v in allv)
    maxf = max(allv) if allv else 0.0
    nc = _num(noncoding) if noncoding is not None else 0.0

    if coding_count >= 4:
        label = "High coding-region transcript-derived mt-variant burden"
        burden = min(1.0, 0.55 + coding_count*0.07 + max(0,maxf-.20))
    elif coding_count >= 1:
        label = "Coding-region transcript-derived mt-variant signal"
        burden = min(0.75, 0.30 + coding_count*0.10 + max(0,maxf-.20))
    elif nc >= .20:
        label = "Non-coding transcript-derived mt-variant signal"
        burden = min(0.55, 0.25 + nc)
    else:
        label = "No ≥20% variants in the current quantified mt panel"
        burden = 0.0

    return {
        "available": True,
        "coding_count20": coding_count,
        "total_count20": total_count,
        "max_fraction": maxf,
        "label": label,
        "burden_score": burden,
    }

def nearest(sample_type, vals):
    ref = df[df["sample_type"] == sample_type].copy()
    rows=[]
    for _, r in ref.iterrows():
        diffs=[]
        for f, uv in vals.items():
            if f not in ref.columns or uv is None:
                continue
            try:
                uv=float(uv)
                if math.isnan(uv):
                    continue
            except:
                continue
            rv=r[f]
            if pd.isna(rv):
                continue
            if f in IMMUNE or f in SCMC or f in MT_VARIANTS:
                d=uv-float(rv)
            else:
                d=robust_z(ref[f],uv)-robust_z(ref[f],rv)
            diffs.append(d*d)
        if diffs:
            dist=math.sqrt(sum(diffs)/len(diffs))
            sim=100/(1+dist)
            rows.append({
                "biological_id":r["biological_id"],
                "sample_id":r["sample_id"],
                "state":r["provisional_state"],
                "archetype":r["provisional_archetype"],
                "distance":dist,
                "similarity":sim,
            })
    return sorted(rows,key=lambda x:x["distance"])

def state_vote(neighbors, sample_type):
    if sample_type=="oocyte":
        return "OOCYTE PROFILE"
    if not neighbors:
        return "UNCLASSIFIED"

    # Similarity-squared weighting gives the nearest molecular anchor
    # appropriately greater influence while retaining information from
    # the next two cohort neighbors. This also guarantees that an exact
    # reference sample reproduces its own provisional cohort state.
    scores={"CLEAN-like":0.0,"INTERMEDIATE":0.0,"DIRTY-like":0.0}
    for n in neighbors[:3]:
        if n["state"] in scores:
            w=max(n["similarity"],1.0) ** 2
            scores[n["state"]] += w
    return max(scores,key=scores.get)

def gmean(a,b):
    return math.sqrt(max(_num(a),0)*max(_num(b),0))

def phenotype_engine(vals, sample_type):
    ref=df[df["sample_type"]==sample_type]
    mp_raw=percentile(ref["mitochondrial_fraction_pct_approx"],vals.get("mitochondrial_fraction_pct_approx"))
    lp_raw=percentile(ref["LINE1_RPM_approx"],vals.get("LINE1_RPM_approx"))
    cp_raw=percentile(ref["chimeric_RPM_approx"],vals.get("chimeric_RPM_approx"))
    irp_raw=percentile(ref["IR_index_approx"],vals.get("IR_index_approx"))
    mp=0 if mp_raw is None else mp_raw
    lp=0 if lp_raw is None else lp_raw
    cp=0 if cp_raw is None else cp_raw
    irp=0 if irp_raw is None else irp_raw
    vs=variant_summary(vals)

    scores={}
    scores["FAS–NFκB-associated"]=gmean(vals.get("FAS"),vals.get("NFKB1"))
    scores["p53–BBC3-associated"]=gmean(vals.get("TP53"),vals.get("BBC3"))
    scores["NLRP3–CASP1-associated"]=gmean(vals.get("NLRP3"),vals.get("CASP1"))
    scores["Nucleic-acid sensor-associated"]=max(
        gmean(vals.get("IFIH1"),vals.get("TLR5")),
        gmean(vals.get("STING1"),vals.get("MDM2")),
        _num(vals.get("TLR3"))
    )

    ir_dev=abs(irp-50)/50
    scmc_pres=np.mean([_num(vals.get(g)) for g in ["NLRP7","KHDC3L","PADI6","BRCA1"]])
    scores["SCMC–DDR preserved"]=float(scmc_pres)

    if sample_type=="embryo":
        scores["Transcriptomic-instability-associated"]=cp/100 if cp>=70 else 0
        scores["RNA-processing deviation"]=ir_dev if ir_dev>=.60 else 0
        scores["Mitochondrial–retrotransposon-associated"]=min(mp,lp)/100 if mp>=70 and lp>=70 else 0

        if vs["available"] and vs["burden_score"]>0:
            scores["Transcript-derived mitochondrial variant-associated"]=vs["burden_score"]

    else:
        mito_retro=min(mp,lp)/100 if mp>=75 and lp>=75 else 0
        scores["Mitochondrial–retrotransposon oocyte profile"]=mito_retro

        partial_scmc=max(_num(vals.get("KHDC3L")),_num(vals.get("PEG3")),_num(vals.get("OOEP")))
        scores["Retrotransposon / partial-SCMC oocyte profile"]=(
            .65*(lp/100)+.35*partial_scmc
            if lp>=65 and partial_scmc>=.35 and mito_retro<.80 else 0
        )

        il18_oas1=gmean(vals.get("IL18"),vals.get("OAS1"))
        scores["IL18/OAS1-associated oocyte profile"]=il18_oas1

        brca_can=_num(vals.get("BRCA1_ENST00000354071_8_FPKM"))
        brca_alt=_num(vals.get("BRCA1_ENST00000634433_2_FPKM"))
        max_can=max(pd.to_numeric(ref["BRCA1_ENST00000354071_8_FPKM"],errors="coerce").fillna(0).max(),1)
        max_alt=max(pd.to_numeric(ref["BRCA1_ENST00000634433_2_FPKM"],errors="coerce").fillna(0).max(),1)
        scores["BRCA1-rich maternal-transcript oocyte profile"]=min(.75*(brca_can/max_can)+.25*(brca_alt/max_alt),1)

        # Integrated P8_O1-like signature only if exact variant panel data are entered.
        # When this integrated signature is present, it supersedes the component IL18/OAS1
        # label to avoid a redundant "Mixed: integrated + component" result.
        if vs["available"] and vs["coding_count20"] is not None and vs["coding_count20"]>=4:
            scores["Mitochondrial variant-burden / IL18-OAS1 oocyte profile"] = (
                .55*vs["burden_score"] + .45*il18_oas1
            )
            scores["IL18/OAS1-associated oocyte profile"] = 0.0

    ranked=sorted(scores.items(),key=lambda kv:kv[1],reverse=True)
    top_name,top_score=ranked[0]
    second_name,second_score=ranked[1]

    if top_score<.35:
        label="Unclassified / low-signal profile"; strength="LOW"; sep=max(0,top_score*100)
    elif top_score<.55:
        label=top_name; strength="MODERATE-LOW"; sep=40+max(0,top_score-second_score)*40
    elif second_score>=.50 and abs(top_score-second_score)<=.12:
        label=f"Mixed: {top_name} + {second_name}"; strength="MODERATE"; sep=55+max(0,top_score-second_score)*50
    else:
        label=top_name; strength="MODERATE-HIGH" if top_score<.80 else "HIGH"; sep=60+min(35,max(0,top_score-second_score)*100)
    sep=max(0,min(95,sep))

    return label,top_score,strength,sep,ranked,{
        "mito_percentile":mp_raw,"line1_percentile":lp_raw,"chimeric_percentile":cp_raw,
        "ir_percentile":irp_raw,"scmc_preservation":scmc_pres*100,"variant":vs
    }

def signal_plot(metrics, vals):
    stress=max(
        gmean(vals.get("FAS"),vals.get("NFKB1")),
        gmean(vals.get("TP53"),vals.get("BBC3")),
        gmean(vals.get("NLRP3"),vals.get("CASP1")),
        gmean(vals.get("IFIH1"),vals.get("TLR5")),
        gmean(vals.get("STING1"),vals.get("MDM2")),
        _num(vals.get("TLR3")),
    )*100
    vburden=metrics["variant"]["burden_score"]*100 if metrics["variant"]["available"] else 0
    labels=["Mito representation","LINE-1 activity","Chimeric reads","IR position","Stress-program","SCMC/DDR preservation","mt-variant burden"]
    data=[
        np.nan if metrics["mito_percentile"] is None else metrics["mito_percentile"],
        np.nan if metrics["line1_percentile"] is None else metrics["line1_percentile"],
        np.nan if metrics["chimeric_percentile"] is None else metrics["chimeric_percentile"],
        np.nan if metrics["ir_percentile"] is None else metrics["ir_percentile"],
        stress,metrics["scmc_preservation"],vburden
    ]
    missing_labels=[labels[i] for i,v in enumerate(data) if isinstance(v,float) and np.isnan(v)]
    display_data=[0 if (isinstance(v,float) and np.isnan(v)) else v for v in data]
    angles=np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist()
    a=angles+[angles[0]]; d=display_data+[display_data[0]]
    fig=plt.figure(figsize=(6.3,6.3),facecolor="#FFFFFF")
    ax=fig.add_subplot(111,polar=True,facecolor="#FFFFFF")
    ax.plot(a,d,linewidth=2.5); ax.fill(a,d,alpha=.16)
    ax.set_xticks(angles); ax.set_xticklabels(labels,fontsize=8.5,color="#344054")
    ax.set_ylim(0,100); ax.set_yticks([20,40,60,80,100]); ax.set_yticklabels(["20","40","60","80","100"],fontsize=8,color="#667085")
    ax.grid(alpha=.18); ax.spines["polar"].set_alpha(.15)
    title="Molecular Signal Map — relative to reference cohort"
    if missing_labels:
        title += "\nMissing: " + ", ".join(missing_labels)
    ax.set_title(title,color="#101828",fontsize=13,pad=25)
    fig.tight_layout()
    return fig

def card(title,value,subtitle="",cls=""):
    return f"""<div class="card {cls}"><div class="eyebrow">{title}</div><div class="card-value">{value}</div><div class="card-sub">{subtitle}</div></div>"""


def _parse_manual_inputs(sample_type,total_reads,detected_genes,mito,line1,chimeric,ir_index,advanced):
    n_i=len(IMMUNE); n_s=len(SCMC); n_b=len(BRCA); n_v=len(MT_VARIANTS); n_iso=len(ISOFORM_COLS)
    ivals=advanced[:n_i]
    svals=advanced[n_i:n_i+n_s]
    bvals=advanced[n_i+n_s:n_i+n_s+n_b]
    vvals=advanced[n_i+n_s+n_b:n_i+n_s+n_b+n_v]
    isovals=advanced[n_i+n_s+n_b+n_v:n_i+n_s+n_b+n_v+n_iso]

    vals={
        "total_reads_approx":total_reads,
        "detected_genes_approx":detected_genes,
        "mitochondrial_fraction_pct_approx":mito,
        "LINE1_RPM_approx":line1,
        "chimeric_RPM_approx":chimeric,
        "IR_index_approx":ir_index,
    }
    vals.update(dict(zip(IMMUNE,ivals)))
    vals.update(dict(zip(SCMC,svals)))
    vals.update(dict(zip(BRCA,bvals)))
    vals.update(dict(zip(MT_VARIANTS,vvals)))
    vals.update(dict(zip(ISOFORM_COLS,isovals)))
    return vals

def _analysis_bundle(sample_type,total_reads,detected_genes,mito,line1,chimeric,ir_index,*advanced):
    vals=_parse_manual_inputs(sample_type,total_reads,detected_genes,mito,line1,chimeric,ir_index,advanced)
    biological_vals = {
        k:v for k,v in vals.items()
        if k not in ["total_reads_approx","detected_genes_approx"] and k not in ISOFORM_COLS
    }
    neighbors=nearest(sample_type,biological_vals)
    if not neighbors:
        return None

    state=state_vote(neighbors,sample_type)
    pheno,pscore,strength,sep,ranked,metrics=phenotype_engine(vals,sample_type)
    tech=technical_context(total_reads,detected_genes)
    iso=isoform_summary(vals,sample_type)
    etis=etis_beta_candidate(vals,sample_type)
    top=neighbors[0]

    domain_df=pd.DataFrame(
        [
            {
                "domain":d["domain"],
                "score_0_100":None if d["score"] is None else round(float(d["score"]),1),
                "included_in_etis":bool(d["included"]),
                "base_weight":round(float(d["base_weight"]),2),
                "candidate_weight":round(float(d["candidate_weight"]),2),
                "effective_weight_pct":None if d["effective_weight_pct"] is None else round(float(d["effective_weight_pct"]),1),
                "beta_rule":d["rule"],
            }
            for d in etis["domains"]
        ]
    )
    top3=pd.DataFrame(neighbors[:3])[["biological_id","sample_id","state","archetype","similarity"]]
    top3["similarity"]=top3["similarity"].round(1)

    return {
        "vals":vals,
        "neighbors":neighbors,
        "state":state,
        "phenotype":pheno,
        "phenotype_score":pscore,
        "phenotype_strength":strength,
        "phenotype_separation":sep,
        "ranked_phenotypes":ranked,
        "metrics":metrics,
        "technical":tech,
        "isoform":iso,
        "etis":etis,
        "top":top,
        "domain_df":domain_df,
        "top3":top3,
    }

def _safe_filename(value):
    value=(value or "sample").strip()
    value=re.sub(r"[^A-Za-z0-9._-]+","_",value)
    return value.strip("._") or "sample"

def _pdf_safe(value):
    s=str(value)
    replacements={
        "–":"-","—":"-","−":"-","·":"-","κ":"k","α":"alpha","β":"beta",
        "≥":">=","≤":"<=","×":"x","“":'"',"”":'"',"’":"'","→":"->",
    }
    for a,b in replacements.items():
        s=s.replace(a,b)
    return s.encode("latin-1","replace").decode("latin-1")

def _fmt_pct(v):
    return "N/A" if v is None else f"{float(v):.0f}th percentile"

def _fmt_score(v):
    return "N/A" if v is None else f"{float(v):.1f}/100"

def _report_key_findings(bundle):
    etis=bundle["etis"]
    tech=bundle["technical"]
    metrics=bundle["metrics"]
    iso=bundle["isoform"]
    vs=metrics["variant"]
    domains=bundle["domain_df"].dropna(subset=["score_0_100"]).copy()

    findings=[]
    if etis["score"] is None:
        findings.append(f"Integrated ETIS beta candidate is not available; domain completeness is {etis['completeness']:.0f}%.")
    else:
        findings.append(f"ETIS beta candidate: {etis['score']:.1f}/100 - {etis['band']}.")

    findings.append(
        f"Dominant molecular signature: {bundle['phenotype']} "
        f"(within-cohort pattern score {bundle['phenotype_score']:.2f})."
    )

    if len(domains):
        low=domains.sort_values("score_0_100").iloc[0]
        high=domains.sort_values("score_0_100",ascending=False).iloc[0]
        findings.append(f"Lowest scored integrity domain: {low['domain']} ({low['score_0_100']:.1f}/100).")
        findings.append(f"Highest scored integrity domain: {high['domain']} ({high['score_0_100']:.1f}/100).")

    findings.append(
        f"Technical context: {tech['status']} "
        f"(reads {_fmt_pct(tech['read_percentile'])}; detected genes {_fmt_pct(tech['gene_percentile'])})."
    )

    if iso["available"]:
        findings.append(
            f"Isoform module: {iso['label']}; closest isoform reference "
            f"{iso['closest_id']} ({iso['closest_sample']}), similarity {iso['similarity']:.1f}%."
        )

    if vs["available"]:
        findings.append(
            f"Mitochondrial variant module: {vs['label']}; "
            f"{vs['total_count20']} quantified variant(s) >=20%, maximum fraction {vs['max_fraction']:.2f}."
        )
    else:
        findings.append("Mitochondrial variant panel was not available for integrated scoring.")

    return findings

def _research_followup(bundle):
    items=[]
    tech=bundle["technical"]
    etis=bundle["etis"]
    domains=bundle["domain_df"].dropna(subset=["score_0_100"]).copy()
    vs=bundle["metrics"]["variant"]

    if tech["status"] in ["LOW COHORT POSITION","CAUTION"]:
        items.append(
            "Interpret low/absent molecular signals cautiously in light of technical context; "
            "orthogonal confirmation or repeat analysis may be useful if material and study design permit."
        )

    scores={r["domain"]:r["score_0_100"] for _,r in domains.iterrows()}
    if scores.get("Transcriptomic stability",100) < 45:
        items.append(
            "Prioritize research validation of transcriptomic-instability signals, including LINE-1/chimeric-read burden "
            "and DDX39B-PEG3 corroboration where applicable."
        )
    if scores.get("RNA processing integrity",100) < 45:
        items.append(
            "Consider targeted validation of RNA-processing features (intron retention and relevant splicing/isoform markers), "
            "including PARP1 context where present."
        )
    if scores.get("Immune / stress integrity",100) < 45:
        items.append(
            "Consider orthogonal validation of the dominant stress/damage programme (for example FAS-NFKB1, TP53-BBC3, "
            "NLRP3-CASP1 or the relevant sensor programme)."
        )
    if scores.get("SCMC / DDR integrity",100) < 45:
        items.append(
            "Consider targeted validation of SCMC/DDR-associated markers such as NLRP7, KHDC3L, PADI6 and BRCA1."
        )
    if scores.get("Mitochondrial integrity",100) < 45 and vs["available"]:
        items.append(
            "Transcript-derived mitochondrial variant signals should be confirmed with an orthogonal DNA-level method before "
            "any interpretation as mtDNA heteroplasmy."
        )
    if not items:
        items.append(
            "No single low-scoring domain triggers a specific follow-up rule in this prototype; interpretation should remain "
            "integrative and cohort-relative."
        )
    return items

def _interpretation_text(bundle):
    etis=bundle["etis"]
    domains=bundle["domain_df"].dropna(subset=["score_0_100"]).copy()
    if len(domains):
        low=domains.sort_values("score_0_100").iloc[0]
        high=domains.sort_values("score_0_100",ascending=False).iloc[0]
        domain_sentence=(
            f"The lowest integrity score is observed in {low['domain']} ({low['score_0_100']:.1f}/100), "
            f"whereas the highest is {high['domain']} ({high['score_0_100']:.1f}/100)."
        )
    else:
        domain_sentence="No integrated domain scores are available."

    if etis["score"] is None:
        etis_sentence=(
            f"An integrated ETIS beta candidate is not reported because the current domain completeness is "
            f"{etis['completeness']:.0f}% or because this material type is not yet supported for integrated ETIS."
        )
    else:
        etis_sentence=(
            f"The sample has a cohort-relative ETIS beta candidate of {etis['score']:.1f}/100, "
            f"classified as {etis['band']}."
        )

    return (
        f"{etis_sentence} The dominant observed molecular signature is {bundle['phenotype']}. "
        f"{domain_sentence} The closest multidimensional cohort reference is "
        f"{bundle['top']['biological_id']} ({bundle['top']['sample_id']}) at "
        f"{bundle['top']['similarity']:.1f}% similarity. These findings describe associative molecular patterns "
        f"within the current reference cohort and do not establish the cause of developmental arrest, embryo/oocyte "
        f"viability, reversibility, diagnosis or treatment response."
    )

def _figure_png_bytes(fig, dpi=170):
    bio=io.BytesIO()
    fig.savefig(bio,format="png",dpi=dpi,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)
    bio.seek(0)
    return bio.getvalue()

def _domain_plot(bundle):
    d=bundle["domain_df"].dropna(subset=["score_0_100"]).copy()
    fig,ax=plt.subplots(figsize=(8.2,4.4))
    if len(d):
        ax.barh(d["domain"],d["score_0_100"])
        ax.set_xlim(0,100)
        ax.set_xlabel("Integrity score (0-100)")
        ax.set_title("ETIS beta candidate domain profile")
        ax.invert_yaxis()
        for y,v in enumerate(d["score_0_100"]):
            ax.text(min(float(v)+1,97),y,f"{float(v):.1f}",va="center",fontsize=9)
    else:
        ax.text(.5,.5,"No domain scores available",ha="center",va="center",transform=ax.transAxes)
        ax.axis("off")
    fig.tight_layout()
    return fig

def _report_payload(sample_label,cycle_id,report_notes,sample_type,total_reads,detected_genes,mito,line1,chimeric,ir_index,*advanced):
    bundle=_analysis_bundle(sample_type,total_reads,detected_genes,mito,line1,chimeric,ir_index,*advanced)
    if bundle is None:
        return None

    generated=datetime.now().astimezone()
    report_id=f"EAMI-{generated.strftime('%Y%m%d-%H%M%S')}"
    return {
        "sample_label":(sample_label or "").strip() or "Unspecified sample",
        "cycle_id":(cycle_id or "").strip(),
        "notes":(report_notes or "").strip(),
        "sample_type":sample_type,
        "generated":generated,
        "report_id":report_id,
        "bundle":bundle,
        "key_findings":_report_key_findings(bundle),
        "interpretation":_interpretation_text(bundle),
        "followup":_research_followup(bundle),
    }

def _html_report(payload, signal_png, domain_png):
    b=payload["bundle"]
    etis=b["etis"]
    tech=b["technical"]
    iso=b["isoform"]
    metrics=b["metrics"]
    vs=metrics["variant"]
    signal64=base64.b64encode(signal_png).decode("ascii")
    domain64=base64.b64encode(domain_png).decode("ascii")

    def esc(x):
        return html.escape(str(x))
    def row(k,v):
        return f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>"

    domain_rows=""
    for _,r in b["domain_df"].iterrows():
        score="N/A" if pd.isna(r["score_0_100"]) else f"{float(r['score_0_100']):.1f}"
        eff="N/A" if pd.isna(r["effective_weight_pct"]) else f"{float(r['effective_weight_pct']):.1f}%"
        domain_rows+=(
            f"<tr><td>{esc(r['domain'])}</td><td>{esc(score)}</td>"
            f"<td>{esc('Yes' if r['included_in_etis'] else 'No')}</td><td>{esc(eff)}</td></tr>"
        )

    findings="".join(f"<li>{esc(x)}</li>" for x in payload["key_findings"])
    followup="".join(f"<li>{esc(x)}</li>" for x in payload["followup"])
    notes_html=f"<p>{esc(payload['notes'])}</p>" if payload["notes"] else "<p>None.</p>"

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>EAMI Molecular Interpretation Report - {esc(payload['sample_label'])}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;margin:0;background:#f4f5f8;color:#20212a;line-height:1.45}}
.page{{max-width:1050px;margin:30px auto;background:white;padding:42px 48px;box-shadow:0 4px 30px rgba(0,0,0,.08)}}
.brand{{color:#A50B5E;font-weight:800;letter-spacing:.04em}}
h1{{margin:.1em 0 .15em;font-size:34px}} h2{{margin-top:30px;border-bottom:2px solid #eee;padding-bottom:8px}}
.meta,.summary{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
.card{{border:1px solid #ddd;border-radius:12px;padding:15px 17px;background:#fafafd}}
.label{{font-size:11px;color:#70727d;text-transform:uppercase;letter-spacing:.08em;font-weight:700}}
.value{{font-size:22px;font-weight:800;margin-top:4px}}
.small{{font-size:12px;color:#70727d;margin-top:5px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} th,td{{border:1px solid #ddd;padding:9px;text-align:left}} th{{background:#f0f1f5}}
figure{{margin:20px 0}} img{{max-width:100%;height:auto;border:1px solid #eee;border-radius:10px}}
.notice{{background:#fff7e6;border-left:4px solid #c98b18;padding:13px 15px;margin:18px 0}}
.ruo{{background:#fbeff5;border-left:4px solid #A50B5E;padding:13px 15px;font-weight:700}}
footer{{margin-top:36px;padding-top:15px;border-top:1px solid #ddd;color:#777;font-size:12px}}
@media print{{body{{background:white}} .page{{box-shadow:none;margin:0;max-width:none}}}}
</style>
</head>
<body><main class="page">
<div class="brand">EAMI</div>
<h1>Molecular Interpretation Report</h1>
<p>Embryo Arrest Molecular Interpreter · v0.8 report layer · ETIS beta candidate v0.2</p>
<div class="ruo">RESEARCH USE ONLY - not for embryo selection, transfer decisions, diagnosis, viability assessment or causal attribution.</div>

<h2>Report metadata</h2>
<table>
{row("Report ID",payload["report_id"])}
{row("Generated",payload["generated"].strftime("%Y-%m-%d %H:%M %Z"))}
{row("Sample label",payload["sample_label"])}
{row("Material type",payload["sample_type"])}
{row("Pseudonymous cycle/subject ID",payload["cycle_id"] or "Not provided")}
</table>

<h2>Executive molecular summary</h2>
<div class="summary">
<div class="card"><div class="label">ETIS beta candidate</div><div class="value">{esc(_fmt_score(etis["score"]))}</div><div class="small">{esc(etis["band"])}</div></div>
<div class="card"><div class="label">ETIS domain completeness</div><div class="value">{etis["completeness"]:.0f}%</div><div class="small">Technical context is separate.</div></div>
<div class="card"><div class="label">Molecular profile class</div><div class="value">{esc(b["state"])}</div><div class="small">Similarity-weighted reference-cohort classification.</div></div>
<div class="card"><div class="label">Dominant molecular signature</div><div class="value" style="font-size:18px">{esc(b["phenotype"])}</div><div class="small">Pattern score {b["phenotype_score"]:.2f}; strength {esc(b["phenotype_strength"])}.</div></div>
</div>

<h2>Key findings</h2><ul>{findings}</ul>
<div class="notice">{esc(payload["interpretation"])}</div>

<h2>ETIS domain profile</h2>
<table><thead><tr><th>Domain</th><th>Score / 100</th><th>Included</th><th>Effective weight</th></tr></thead><tbody>{domain_rows}</tbody></table>
<figure><img src="data:image/png;base64,{domain64}"><figcaption>ETIS beta candidate domain profile.</figcaption></figure>

<h2>Molecular signal map</h2>
<figure><img src="data:image/png;base64,{signal64}"><figcaption>Cohort-relative molecular signal map. Missing values are explicitly marked by the analytical engine.</figcaption></figure>

<h2>Technical context and cohort anchoring</h2>
<table>
{row("Technical context",tech["status"])}
{row("Read-depth position",_fmt_pct(tech["read_percentile"]))}
{row("Detected-gene position",_fmt_pct(tech["gene_percentile"]))}
{row("Closest reference",f"{b['top']['biological_id']} ({b['top']['sample_id']})")}
{row("Closest-reference similarity",f"{b['top']['similarity']:.1f}%")}
</table>

<h2>Module detail</h2>
<table>
{row("Mitochondrial fraction",_fmt_pct(metrics["mito_percentile"]))}
{row("LINE-1",_fmt_pct(metrics["line1_percentile"]))}
{row("Chimeric reads",_fmt_pct(metrics["chimeric_percentile"]))}
{row("Intron retention",_fmt_pct(metrics["ir_percentile"]))}
{row("SCMC/DDR preservation signal",f"{metrics['scmc_preservation']:.0f}/100")}
{row("Isoform pattern",iso["label"] if iso["available"] else "Not assessed")}
{row("Mitochondrial variant module",vs["label"] if vs["available"] else "Not available")}
</table>

<h2>Research follow-up considerations</h2><ul>{followup}</ul>

<h2>Research notes</h2>{notes_html}

<h2>Limitations</h2>
<ul>
<li>The current reference system is based on the supplied 16-sample cohort and is not an external normal/viable embryo reference.</li>
<li>ETIS beta candidate rules were refined after reviewing discordant cases in the same development cohort; development-set agreement is not independent validation.</li>
<li>Some reference values were reconstructed or digitized from supplied figures; the 25-transcript isoform panel is approximate.</li>
<li>Transcript-derived mitochondrial variant fractions must not be interpreted as direct absolute mtDNA heteroplasmy measurements.</li>
<li>High or low chrM representation is treated as context, not as a monotonic integrity marker.</li>
<li>EAMI does not establish causality, viability, reversibility, diagnosis, treatment response or probability of implantation.</li>
</ul>
<footer>{esc(payload["report_id"])} · EAMI v0.8 · ETIS beta candidate v0.2 · Research Use Only</footer>
</main></body></html>"""

def _pdf_report(path, payload, signal_png, domain_png):
    b=payload["bundle"]
    etis=b["etis"]
    tech=b["technical"]
    iso=b["isoform"]
    metrics=b["metrics"]
    vs=metrics["variant"]

    styles=getSampleStyleSheet()
    title_style=ParagraphStyle(
        "EAMITitle",parent=styles["Title"],fontName="Helvetica-Bold",
        fontSize=22,leading=26,textColor=colors.HexColor("#A50B5E"),spaceAfter=5
    )
    h1=ParagraphStyle(
        "EAMIH1",parent=styles["Heading2"],fontName="Helvetica-Bold",
        fontSize=14,leading=17,textColor=colors.HexColor("#272832"),spaceBefore=10,spaceAfter=7
    )
    body=ParagraphStyle(
        "EAMIBody",parent=styles["BodyText"],fontName="Helvetica",
        fontSize=9.5,leading=13,spaceAfter=6
    )
    small=ParagraphStyle(
        "EAMISmall",parent=body,fontSize=8,leading=10,textColor=colors.HexColor("#666666")
    )
    notice=ParagraphStyle(
        "EAMINotice",parent=body,fontName="Helvetica-Bold",backColor=colors.HexColor("#FBEFF5"),
        borderColor=colors.HexColor("#A50B5E"),borderWidth=0.8,borderPadding=7,spaceBefore=5,spaceAfter=10
    )

    doc=SimpleDocTemplate(
        str(path),pagesize=A4,rightMargin=15*mm,leftMargin=15*mm,topMargin=15*mm,bottomMargin=15*mm,
        title=f"EAMI Molecular Interpretation Report - {payload['sample_label']}",
        author="EAMI Research Prototype"
    )
    story=[]
    story.append(Paragraph("EAMI",title_style))
    story.append(Paragraph("Molecular Interpretation Report",ParagraphStyle(
        "Subtitle",parent=styles["Heading1"],fontName="Helvetica-Bold",fontSize=17,leading=21,spaceAfter=3
    )))
    story.append(Paragraph("Embryo Arrest Molecular Interpreter - v0.8 report layer - ETIS beta candidate v0.2",small))
    story.append(Spacer(1,3*mm))
    story.append(Paragraph(
        "RESEARCH USE ONLY - not for embryo selection, transfer decisions, diagnosis, viability assessment or causal attribution.",
        notice
    ))

    def P(v, style=body):
        return Paragraph(_pdf_safe(v),style)

    story.append(P("Report metadata",h1))
    meta=[
        ["Report ID",payload["report_id"]],
        ["Generated",payload["generated"].strftime("%Y-%m-%d %H:%M %Z")],
        ["Sample label",payload["sample_label"]],
        ["Material type",payload["sample_type"]],
        ["Pseudonymous cycle/subject ID",payload["cycle_id"] or "Not provided"],
    ]
    mt=Table([[P(a,small),P(b,body)] for a,b in meta],colWidths=[52*mm,118*mm])
    mt.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#D0D0D5")),
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F3F3F6")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(mt)

    story.append(P("Executive molecular summary",h1))
    score_text=_fmt_score(etis["score"])
    summary=[
        ["ETIS beta candidate",score_text],
        ["ETIS interpretation band",etis["band"]],
        ["ETIS domain completeness",f"{etis['completeness']:.0f}%"],
        ["Molecular profile class",b["state"]],
        ["Dominant molecular signature",b["phenotype"]],
        ["Closest cohort reference",f"{b['top']['biological_id']} ({b['top']['sample_id']}) - {b['top']['similarity']:.1f}%"],
        ["Technical context",tech["status"]],
    ]
    st=Table([[P(a,small),P(v,body)] for a,v in summary],colWidths=[55*mm,115*mm])
    st.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#D0D0D5")),
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F3F3F6")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(st)
    story.append(Spacer(1,3*mm))

    story.append(P("Key findings",h1))
    for item in payload["key_findings"]:
        story.append(P(f"- {item}",body))
    story.append(Paragraph(_pdf_safe(payload["interpretation"]),notice))

    story.append(PageBreak())
    story.append(P("ETIS domain profile",h1))
    domain_data=[[P("Domain",small),P("Score",small),P("Included",small),P("Effective weight",small)]]
    for _,r in b["domain_df"].iterrows():
        score="N/A" if pd.isna(r["score_0_100"]) else f"{float(r['score_0_100']):.1f}"
        eff="N/A" if pd.isna(r["effective_weight_pct"]) else f"{float(r['effective_weight_pct']):.1f}%"
        domain_data.append([P(r["domain"],body),P(score,body),P("Yes" if r["included_in_etis"] else "No",body),P(eff,body)])
    dt=Table(domain_data,colWidths=[70*mm,25*mm,25*mm,40*mm],repeatRows=1)
    dt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EDEDF2")),
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#D0D0D5")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story.append(dt)
    story.append(Spacer(1,4*mm))
    story.append(RLImage(io.BytesIO(domain_png),width=165*mm,height=88*mm))

    story.append(PageBreak())
    story.append(P("Molecular signal map",h1))
    story.append(RLImage(io.BytesIO(signal_png),width=145*mm,height=145*mm))
    story.append(P(
        "The molecular signal map is cohort-relative. Missing inputs are handled explicitly by the analytical engine and should not be interpreted as biological zeros.",
        small
    ))

    story.append(P("Technical context and module detail",h1))
    details=[
        ["Read-depth position",_fmt_pct(tech["read_percentile"])],
        ["Detected-gene position",_fmt_pct(tech["gene_percentile"])],
        ["Mitochondrial fraction",_fmt_pct(metrics["mito_percentile"])],
        ["LINE-1",_fmt_pct(metrics["line1_percentile"])],
        ["Chimeric reads",_fmt_pct(metrics["chimeric_percentile"])],
        ["Intron retention",_fmt_pct(metrics["ir_percentile"])],
        ["SCMC/DDR preservation",f"{metrics['scmc_preservation']:.0f}/100"],
        ["Isoform pattern",iso["label"] if iso["available"] else "Not assessed"],
        ["Mitochondrial variant module",vs["label"] if vs["available"] else "Not available"],
    ]
    dtab=Table([[P(a,small),P(v,body)] for a,v in details],colWidths=[58*mm,112*mm])
    dtab.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#D0D0D5")),
        ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F3F3F6")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(dtab)

    story.append(PageBreak())
    story.append(P("Research follow-up considerations",h1))
    for item in payload["followup"]:
        story.append(P(f"- {item}",body))

    story.append(P("Research notes",h1))
    story.append(P(payload["notes"] or "None.",body))

    story.append(P("Limitations",h1))
    limitations=[
        "The current reference system is based on the supplied 16-sample cohort and is not an external normal/viable embryo reference.",
        "ETIS beta candidate rules were refined after reviewing discordant cases in the same development cohort; development-set agreement is not independent validation.",
        "Some reference values were reconstructed or digitized from supplied figures; the 25-transcript isoform panel is approximate.",
        "Transcript-derived mitochondrial variant fractions must not be interpreted as direct absolute mtDNA heteroplasmy measurements.",
        "High or low chrM representation is treated as context, not as a monotonic integrity marker.",
        "EAMI does not establish causality, viability, reversibility, diagnosis, treatment response or probability of implantation.",
    ]
    for item in limitations:
        story.append(P(f"- {item}",body))

    story.append(Spacer(1,4*mm))
    story.append(P(f"{payload['report_id']} - EAMI v0.8.1 - ETIS beta candidate v0.2 - Research Use Only",small))

    def footer(canvas,doc):
        canvas.saveState()
        canvas.setFont("Helvetica",7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(15*mm,8*mm,_pdf_safe(payload["report_id"]))
        canvas.drawRightString(A4[0]-15*mm,8*mm,f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story,onFirstPage=footer,onLaterPages=footer)

def generate_report(sample_label,cycle_id,report_notes,sample_type,total_reads,detected_genes,mito,line1,chimeric,ir_index,*advanced):
    payload=_report_payload(
        sample_label,cycle_id,report_notes,sample_type,total_reads,detected_genes,mito,line1,chimeric,ir_index,*advanced
    )
    if payload is None:
        return "Report could not be generated because no cohort neighbor could be calculated.",None,None

    stamp=payload["generated"].strftime("%Y%m%d_%H%M%S")
    stem=f"EAMI_{_safe_filename(payload['sample_label'])}_{stamp}"
    html_path=REPORT_DIR/f"{stem}.html"
    pdf_path=REPORT_DIR/f"{stem}.pdf"

    signal_png=_figure_png_bytes(signal_plot(payload["bundle"]["metrics"],payload["bundle"]["vals"]))
    domain_png=_figure_png_bytes(_domain_plot(payload["bundle"]))

    html_path.write_text(_html_report(payload,signal_png,domain_png),encoding="utf-8")
    _pdf_report(pdf_path,payload,signal_png,domain_png)

    etis=payload["bundle"]["etis"]
    summary=(
        f"### Report generated\n"
        f"- **Sample:** {payload['sample_label']}\n"
        f"- **ETIS beta candidate:** {_fmt_score(etis['score'])}\n"
        f"- **ETIS interpretation band:** {etis['band']}\n"
        f"- **Dominant molecular signature:** {payload['bundle']['phenotype']}\n"
        f"- **Report ID:** {payload['report_id']}\n\n"
        f"Both files are self-contained outputs from the current research prototype."
    )
    return summary,str(html_path),str(pdf_path)

def analyze(sample_type,total_reads,detected_genes,mito,line1,chimeric,ir_index,*advanced):
    n_i=len(IMMUNE); n_s=len(SCMC); n_b=len(BRCA); n_v=len(MT_VARIANTS); n_iso=len(ISOFORM_COLS)
    ivals=advanced[:n_i]
    svals=advanced[n_i:n_i+n_s]
    bvals=advanced[n_i+n_s:n_i+n_s+n_b]
    vvals=advanced[n_i+n_s+n_b:n_i+n_s+n_b+n_v]
    isovals=advanced[n_i+n_s+n_b+n_v:n_i+n_s+n_b+n_v+n_iso]

    vals={
        "total_reads_approx":total_reads,
        "detected_genes_approx":detected_genes,
        "mitochondrial_fraction_pct_approx":mito,
        "LINE1_RPM_approx":line1,
        "chimeric_RPM_approx":chimeric,
        "IR_index_approx":ir_index,
    }
    vals.update(dict(zip(IMMUNE,ivals)))
    vals.update(dict(zip(SCMC,svals)))
    vals.update(dict(zip(BRCA,bvals)))
    vals.update(dict(zip(MT_VARIANTS,vvals)))
    vals.update(dict(zip(ISOFORM_COLS,isovals)))

    biological_vals = {k:v for k,v in vals.items() if k not in ["total_reads_approx","detected_genes_approx"] and k not in ISOFORM_COLS}
    neighbors=nearest(sample_type,biological_vals)
    if not neighbors:
        return "","","","","","", "", "", pd.DataFrame(), pd.DataFrame(), None, "Insufficient values."

    state=state_vote(neighbors,sample_type)
    pheno,pscore,strength,sep,ranked,metrics=phenotype_engine(vals,sample_type)
    tech=technical_context(total_reads,detected_genes)
    iso=isoform_summary(vals,sample_type)
    etis=etis_beta_candidate(vals,sample_type)
    top=neighbors[0]

    state_cls={"CLEAN-like":"clean","INTERMEDIATE":"intermediate","DIRTY-like":"dirty","OOCYTE PROFILE":"oocyte"}.get(state,"")
    state_html=card("MOLECULAR PROFILE CLASS",state,"Similarity-weighted classification from the closest reference profiles.",state_cls)
    pheno_html=card("DOMINANT MOLECULAR SIGNATURE",pheno,f"Within-cohort pattern score {pscore:.2f}; pattern strength {strength}; separation {sep:.0f}/100.")
    ref_html=card("CLOSEST COHORT REFERENCE",f"{top['biological_id']} <span class='muted'>({top['sample_id']})</span>",top["archetype"])
    sim_html=card("CLOSEST-REFERENCE SIMILARITY",f"{top['similarity']:.1f}%","Multidimensional cohort similarity.")

    def fp(v):
        return "N/A" if v is None else f"{v:.0f}th percentile"

    tech_html=card(
        "TECHNICAL CONTEXT",
        tech["status"],
        f"Reads: {fp(tech['read_percentile'])} · Detected genes: {fp(tech['gene_percentile'])}. {tech['note']}"
    )

    if iso["available"]:
        iso_sub=f"Closest isoform reference: {iso['closest_id']} ({iso['closest_sample']}) · similarity {iso['similarity']:.1f}% · active panel entries ≥0.5: {iso['active_count']}"
        iso_html=card("ISOFORM PATTERN",iso["label"],iso_sub)
    else:
        iso_html=card("ISOFORM PATTERN","NOT ASSESSED","No isoform-panel values were entered.")

    if etis["score"] is None:
        etis_value="N/A"
    else:
        etis_value=f"{etis['score']:.1f} / 100"
    etis_html=card(
        "ETIS BETA CANDIDATE",
        etis_value,
        f"{etis['band']} · {etis['band_note']}",
        "clean" if etis["score"] is not None and etis["score"] >= 85 else (
            "dirty" if etis["score"] is not None and etis["score"] < 45 else "intermediate"
        )
    )
    completeness_html=card(
        "ETIS DOMAIN COMPLETENESS",
        f"{etis['completeness']:.0f}%",
        "Percentage of the five ETIS-beta biological domains with a calculable score. Technical context is separate."
    )

    domain_df=pd.DataFrame(
        [
            {
                "domain":d["domain"],
                "score_0_100":None if d["score"] is None else round(float(d["score"]),1),
                "included_in_etis":bool(d["included"]),
                "base_weight":round(float(d["base_weight"]),2),
                "candidate_weight":round(float(d["candidate_weight"]),2),
                "effective_weight_pct":None if d["effective_weight_pct"] is None else round(float(d["effective_weight_pct"]),1),
                "beta_rule":d["rule"],
            }
            for d in etis["domains"]
        ]
    )

    top3=pd.DataFrame(neighbors[:3])[["biological_id","sample_id","state","archetype","similarity"]]
    top3["similarity"]=top3["similarity"].round(1)

    ranked_text="\n".join([f"- **{name}:** {score:.2f}" for name,score in ranked[:7]])
    vs=metrics["variant"]
    if not vs["available"]:
        variant_text="Variant panel: **not entered / not available**."
    else:
        variant_text=(
            f"Variant panel: **{vs['label']}**  \n"
            f"- Coding variants ≥20%: **{vs['coding_count20']}**  \n"
            f"- Total variants ≥20%: **{vs['total_count20']}**  \n"
            f"- Maximum transcript-derived variant fraction: **{vs['max_fraction']:.2f}**"
        )

    def fmt_pct(v):
        return "**N/A — not available**" if v is None else f"**{v:.0f}th percentile**"

    if not iso["available"]:
        isoform_text="Not entered."
    else:
        mtnd_txt="N/A" if iso["mtnd_block"] is None else f"{iso['mtnd_block']:.2f}"
        maternal_txt="N/A" if iso["maternal_block"] is None else f"{iso['maternal_block']:.2f}"
        isoform_text=(
            f"{iso['label']}  \n"
            f"- Closest isoform reference: **{iso['closest_id']} ({iso['closest_sample']})**  \n"
            f"- Isoform-pattern similarity: **{iso['similarity']:.1f}%**  \n"
            f"- Active panel entries ≥0.5: **{iso['active_count']} / 25**  \n"
            f"- MTND-pseudogene block mean: **{mtnd_txt}**  \n"
            f"- Maternal/transition lower-block mean: **{maternal_txt}**"
        )

    explanation=f"""
### Explainability

**Core cohort positions**
- Mitochondrial fraction: {fmt_pct(metrics['mito_percentile'])}
- LINE-1: {fmt_pct(metrics['line1_percentile'])}
- Chimeric reads: {fmt_pct(metrics['chimeric_percentile'])}
- Intron retention: {fmt_pct(metrics['ir_percentile'])}
- SCMC/DDR preservation signal: **{metrics['scmc_preservation']:.0f}/100**

**Mitochondrial variant module**  
{variant_text}

**Isoform-switching module**  
{isoform_text}

**ETIS beta candidate**
- Integrated score: **{("N/A" if etis["score"] is None else f"{etis['score']:.1f}/100")}**
- ETIS interpretation band: **{etis["band"]}**
- ETIS domain completeness: **{etis["completeness"]:.0f}%**
- DDX39B–PEG3 corroboration: **{etis["details"]["ddx39b_peg3"]:.2f}**
- PARP1 + extreme IR corroboration: **{("YES" if etis["details"]["parp1_ir_corroboration"] else "NO")}**
- Core damage programme: **{etis["details"]["core_damage"]:.2f}**
- Sensor/adaptive programme: **{etis["details"]["sensor_activity"]:.2f}** (penalty factor {etis["details"]["sensor_penalty_factor"]:.2f})
- ETIS beta candidate does **not** directly score technical context, chrM representation, BRCA1 isoform identity, or the 25-transcript isoform pattern.

**Phenotype-rule scores**
{ranked_text}

**Method note:** mitochondrial variant fractions in the current module are transcript-derived ONT/WTA signals and must not be presented as direct absolute mtDNA heteroplasmy measurements.

**Interpretation note:** EAMI describes within-cohort molecular patterns. It does not establish causality, viability, reversibility, or diagnosis.
"""
    return state_html,pheno_html,ref_html,sim_html,tech_html,iso_html,etis_html,completeness_html,domain_df,top3,signal_plot(metrics,vals),explanation

def filter_reference_samples(material_type):
    """Filter reference demo samples by biological material type."""
    choices = df.loc[
        df["sample_type"].eq(material_type),
        "sample_id"
    ].tolist()

    return gr.update(
        choices=choices,
        value=choices[0] if choices else None
    )


def load_reference(sample_id):
    if not sample_id:
        return [gr.update()]*(7+len(IMMUNE)+len(SCMC)+len(BRCA)+len(MT_VARIANTS)+len(ISOFORM_COLS))
    r=df[df["sample_id"]==sample_id].iloc[0]
    vals=[
        r["sample_type"],
        None if pd.isna(r["total_reads_approx"]) else float(r["total_reads_approx"]),
        None if pd.isna(r["detected_genes_approx"]) else float(r["detected_genes_approx"]),
        None if pd.isna(r["mitochondrial_fraction_pct_approx"]) else float(r["mitochondrial_fraction_pct_approx"]),
        float(r["LINE1_RPM_approx"]),float(r["chimeric_RPM_approx"]),float(r["IR_index_approx"]),
    ]
    vals += [float(r[g]) for g in IMMUNE]
    vals += [float(r[g]) for g in SCMC]
    vals += [float(r[g]) for g in BRCA]
    vals += [None if pd.isna(r[g]) else float(r[g]) for g in MT_VARIANTS]
    vals += [None if pd.isna(r[g]) else float(r[g]) for g in ISOFORM_COLS]
    return vals


def _score_class(score):
    if score is None:
        return "neutral"
    if score >= 85:
        return "good"
    if score < 45:
        return "bad"
    return "mid"

def _ui_card(label, value, subtitle="", cls="neutral", compact=False):
    c = " ui-card compact" if compact else " ui-card"
    return (
        f'<div class="{c.strip()} {cls}">'
        f'<div class="ui-eyebrow">{html.escape(str(label))}</div>'
        f'<div class="ui-value">{value}</div>'
        f'<div class="ui-sub">{html.escape(str(subtitle))}</div>'
        f'</div>'
    )

DOMAIN_HELP_TEXT = {
    "Mitochondrial integrity": (
        "Based on transcript-derived mitochondrial variant burden. "
        "chrM representation is contextual and is not directly scored "
        "as mitochondrial integrity."
    ),

    "Transcriptomic stability": (
        "Cohort-relative assessment of LINE-1 activity and chimeric-read burden, "
        "with DDX39B-PEG3 as corroborating evidence of transcriptomic instability."
    ),

    "RNA processing integrity": (
        "Based primarily on the cohort-relative position of intron retention. "
        "Extreme intron-retention patterns can be corroborated by PARP1. "
        "Isoform information remains contextual in the current beta model."
    ),

    "Immune / stress integrity": (
        "Integrates FAS-NFKB1, TP53-BBC3 and NLRP3-CASP1 core stress programmes, "
        "together with more softly weighted sensor-associated signals such as "
        "IFIH1/TLR5 and STING1/MDM2."
    ),

    "SCMC / DDR integrity": (
        "Represents the mean preservation signal of NLRP7, KHDC3L, "
        "PADI6 and BRCA1 in the current SCMC/DDR research domain."
    ),
}


def _domain_profile_html(bundle):
    rows=[]
    for _, r in bundle["domain_df"].iterrows():
        name=str(r["domain"])
        score=None if pd.isna(r["score_0_100"]) else float(r["score_0_100"])
        included=bool(r["included_in_etis"])
        if score is None:
            score_txt="N/A"
            width=0
            cls="neutral"
        else:
            score_txt=f"{score:.1f}"
            width=max(0,min(100,score))
            cls=_score_class(score)

        tag = "included" if included else "context"

        help_text = DOMAIN_HELP_TEXT.get(
            name,
            "Research-domain score used in the current EAMI interpretation."
        )
        safe_help = html.escape(help_text, quote=True)

        rows.append(
            f'<div class="domain-row">'
            f'  <div class="domain-head">'
            f'    <span class="domain-name-wrap">'
            f'      <span>{html.escape(name)}</span>'
            f'      <span class="domain-help" tabindex="0" '
            f'            data-tip="{safe_help}">i</span>'
            f'    </span>'
            f'    <span class="domain-score {cls}">{score_txt}</span>'
            f'  </div>'
            f'  <div class="domain-track"><div class="domain-fill {cls}" style="width:{width:.1f}%"></div></div>'
            f'  <div class="domain-foot">{tag}</div>'
            f'</div>'
        )
    return '<div class="domain-grid">' + "".join(rows) + "</div>"

def _key_findings_md(bundle):
    etis = bundle["etis"]
    tech = bundle["technical"]
    domains = bundle["domain_df"].dropna(subset=["score_0_100"]).copy()
    top = bundle["top"]

    # ETIS
    if etis["score"] is None:
        etis_line = f"N/A · {etis['band']}"
    else:
        etis_line = f"{etis['score']:.1f}/100 · {etis['band']}"

    # Lowest / highest integrity domains
    if len(domains):
        low = domains.sort_values("score_0_100").iloc[0]
        high = domains.sort_values("score_0_100", ascending=False).iloc[0]

        range_line = (
            f"{low['domain']} {low['score_0_100']:.1f} → "
            f"{high['domain']} {high['score_0_100']:.1f}"
        )
    else:
        range_line = "No integrated domain scores available"

    # Technical context
    tech_line = (
        f"{tech['status']} · "
        f"reads {_fmt_pct(tech['read_percentile'])} · "
        f"genes {_fmt_pct(tech['gene_percentile'])}"
    )

    # Closest cohort reference
    reference_line = (
        f"{top['biological_id']} ({top['sample_id']}) · "
        f"{top['similarity']:.1f}% similarity"
    )

    # Short interpretation
    if etis["score"] is None:
        interpretation = (
            f"Dominant molecular pattern: **{bundle['phenotype']}**. "
            "Integrated ETIS interpretation is not available for the current data."
        )
    else:
        interpretation = (
            f"Overall profile: **{etis['band']}**, with a dominant "
            f"**{bundle['phenotype']}** molecular signature. "
            "Interpretation is cohort-relative and research-only."
        )

    return f"""### Key findings

- **ETIS:** {etis_line}
- **Dominant signature:** {bundle['phenotype']} · score {bundle['phenotype_score']:.2f}
- **Integrity range:** {range_line}
- **Technical context:** {tech_line}
- **Closest reference:** {reference_line}

### Interpretation

{interpretation}
"""


def analyze_ui(sample_type,total_reads,detected_genes,mito,line1,chimeric,ir_index,*advanced):
    b=_analysis_bundle(sample_type,total_reads,detected_genes,mito,line1,chimeric,ir_index,*advanced)
    if b is None:
        empty=_ui_card("ANALYSIS","Insufficient values","No cohort neighbor could be calculated.","neutral")
        return empty,empty,empty,empty,empty,empty,empty,"",pd.DataFrame(),pd.DataFrame(),None,"Insufficient values."

    etis=b["etis"]
    state=b["state"]
    pheno=b["phenotype"]
    tech=b["technical"]
    iso=b["isoform"]
    top=b["top"]

    score=etis["score"]
    etis_value="N/A" if score is None else f"{score:.1f}<span class='unit'>/100</span>"
    etis_card=_ui_card(
        "ETIS BETA CANDIDATE",
        etis_value,
        etis["band"],
        _score_class(score),
    )

    state_cls={"CLEAN-like":"good","INTERMEDIATE":"mid","DIRTY-like":"bad","OOCYTE PROFILE":"oocyte"}.get(state,"neutral")
    state_card=_ui_card(
        "MOLECULAR PROFILE CLASS",
        html.escape(state),
        "Similarity-weighted classification against the reference cohort",
        state_cls,
    )

    pheno_card=_ui_card(
        "DOMINANT MOLECULAR SIGNATURE",
        html.escape(pheno),
        f"Pattern score {b['phenotype_score']:.2f} · {b['phenotype_strength']}",
        "neutral",
    )

    comp=etis["completeness"]
    completeness_card=_ui_card(
        "ETIS DOMAIN COMPLETENESS",
        f"{comp:.0f}<span class='unit'>%</span>",
        "Calculable biological domains used in ETIS",
        "good" if comp >= 80 else ("mid" if comp >= 60 else "bad"),
    )

    ref_card=_ui_card(
        "CLOSEST COHORT REFERENCE",
        f"{html.escape(str(top['biological_id']))} <span class='muted'>({html.escape(str(top['sample_id']))})</span>",
        f"{top['similarity']:.1f}% similarity · {top['archetype']}",
        "neutral",
        compact=True,
    )

    tech_card=_ui_card(
        "TECHNICAL CONTEXT",
        html.escape(tech["status"]),
        f"Reads {_fmt_pct(tech['read_percentile'])} · genes {_fmt_pct(tech['gene_percentile'])}",
        "mid" if tech["status"]=="CAUTION" else ("bad" if tech["status"]=="LOW COHORT POSITION" else "neutral"),
        compact=True,
    )

    iso_value=iso["label"] if iso["available"] else "NOT ASSESSED"
    iso_sub=(
        f"{iso['similarity']:.1f}% to {iso['closest_id']} ({iso['closest_sample']})"
        if iso["available"] else "No isoform panel entered"
    )
    iso_card=_ui_card("ISOFORM PATTERN",html.escape(iso_value),iso_sub,"neutral",compact=True)

    domain_html=_domain_profile_html(b)
    findings=_key_findings_md(b)

    # Keep the full existing explainability content but collapse it lower in the UI.
    metrics=b["metrics"]
    vs=metrics["variant"]
    ranked=b["ranked_phenotypes"]

    def fmt_pct(v):
        return "**N/A — not available**" if v is None else f"**{v:.0f}th percentile**"

    variant_text="Variant panel: **not entered / not available**."
    if vs["available"]:
        variant_text=(
            f"Variant panel: **{vs['label']}**  \n"
            f"- Coding variants ≥20%: **{vs['coding_count20']}**  \n"
            f"- Total variants ≥20%: **{vs['total_count20']}**  \n"
            f"- Maximum transcript-derived variant fraction: **{vs['max_fraction']:.2f}**"
        )

    if not iso["available"]:
        isoform_text="Not entered."
    else:
        mtnd_txt="N/A" if iso["mtnd_block"] is None else f"{iso['mtnd_block']:.2f}"
        maternal_txt="N/A" if iso["maternal_block"] is None else f"{iso['maternal_block']:.2f}"
        isoform_text=(
            f"{iso['label']}  \n"
            f"- Closest isoform reference: **{iso['closest_id']} ({iso['closest_sample']})**  \n"
            f"- Isoform-pattern similarity: **{iso['similarity']:.1f}%**  \n"
            f"- Active panel entries ≥0.5: **{iso['active_count']} / 25**  \n"
            f"- MTND-pseudogene block mean: **{mtnd_txt}**  \n"
            f"- Maternal/transition lower-block mean: **{maternal_txt}**"
        )

    ranked_text="\n".join([f"- **{name}:** {score:.2f}" for name,score in ranked[:7]])
    detailed=f"""
### Core cohort positions
- Mitochondrial fraction: {fmt_pct(metrics['mito_percentile'])}
- LINE-1: {fmt_pct(metrics['line1_percentile'])}
- Chimeric reads: {fmt_pct(metrics['chimeric_percentile'])}
- Intron retention: {fmt_pct(metrics['ir_percentile'])}
- SCMC/DDR preservation signal: **{metrics['scmc_preservation']:.0f}/100**

### Mitochondrial variant module
{variant_text}

### Isoform-switching module
{isoform_text}

### ETIS beta candidate
- Integrated score: **{("N/A" if score is None else f"{score:.1f}/100")}**
- ETIS interpretation band: **{etis["band"]}**
- ETIS domain completeness: **{etis["completeness"]:.0f}%**
- DDX39B–PEG3 corroboration: **{etis["details"]["ddx39b_peg3"]:.2f}**
- PARP1 + extreme IR corroboration: **{("YES" if etis["details"]["parp1_ir_corroboration"] else "NO")}**
- Core damage programme: **{etis["details"]["core_damage"]:.2f}**
- Sensor/adaptive programme: **{etis["details"]["sensor_activity"]:.2f}**

### Phenotype-rule scores
{ranked_text}

**Method note:** mitochondrial variant fractions are transcript-derived ONT/WTA signals and are not direct absolute mtDNA heteroplasmy measurements.

**Interpretation note:** EAMI describes cohort-relative molecular patterns. It does not establish causality, viability, reversibility or diagnosis.
"""

    return (
        etis_card,state_card,pheno_card,completeness_card,
        ref_card,tech_card,iso_card,domain_html,
        b["domain_df"],b["top3"],signal_plot(metrics,b["vals"]),findings + "\n\n" + detailed
    )

def generate_report_ui(sample_label,cycle_id,report_notes,sample_type,total_reads,detected_genes,mito,line1,chimeric,ir_index,*advanced):
    summary,html_path,pdf_path=generate_report(
        sample_label,cycle_id,report_notes,sample_type,total_reads,detected_genes,mito,line1,chimeric,ir_index,*advanced
    )
    if not html_path or not pdf_path:
        return summary,gr.update(value=None,interactive=False),gr.update(value=None,interactive=False)
    return (
        summary,
        gr.update(value=html_path,interactive=True),
        gr.update(value=pdf_path,interactive=True),
    )

CSS="""
:root {
  --eami-accent:#A50B5E;
  --eami-bg:#F7F8FA;
  --eami-panel:#FFFFFF;
  --eami-panel-2:#F2F4F7;
  --eami-border:#DDE1E7;
  --eami-muted:#667085;
  --eami-good:#2F855A;
  --eami-mid:#B7791F;
  --eami-bad:#C2414B;
}

body,
.gradio-container {
  background:var(--eami-bg) !important;
  color:#1F2937 !important;
}

.gradio-container {
  max-width:1280px !important;
  margin:auto !important;
  padding-bottom:90px !important;
}

#hero {
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  gap:18px;
  margin:10px 0 18px;
}
#hero h1 {
  font-size:42px;
  line-height:1;
  margin:0 0 7px;
  letter-spacing:-0.035em;
}
#hero .sub {font-size:16px;color:#a9aab4;}
#hero .tag {
  display:inline-flex;
  padding:6px 10px;
  border:1px solid #3c3e49;
  border-radius:999px;
  color:#bcbec8;
  background:#17181d;
  font-size:11px;
  white-space:nowrap;
}

.section-title {
  margin:25px 0 10px;
  font-size:13px;
  text-transform:uppercase;
  letter-spacing:.10em;
  color:#8f919d;
  font-weight:800;
}

.ui-card {
  border:1px solid var(--eami-border);
  border-radius:16px;
  padding:17px 18px;
  min-height:124px;
  background:#FFFFFF;
  box-shadow:0 4px 18px rgba(20,30,50,.06);
}
.ui-card.compact {
  min-height:94px;
  padding:14px 16px;
}
.ui-card.good {border-color:rgba(84,168,123,.75);}
.ui-card.mid {border-color:rgba(201,154,56,.78);}
.ui-card.bad {border-color:rgba(181,85,97,.82);}
.ui-card.oocyte {border-color:#5879b4;}
.ui-eyebrow {
  font-size:10px;
  font-weight:800;
  letter-spacing:.10em;
  color:#9699a6;
  margin-bottom:8px;
}
.ui-value {
  font-size:24px;
  line-height:1.12;
  font-weight:800;
  letter-spacing:-.02em;
}
.compact .ui-value {font-size:18px;}
.ui-sub {
  color:#979aa7;
  font-size:11.5px;
  line-height:1.4;
  margin-top:7px;
}
.unit {font-size:.55em;color:#aaaeb8;font-weight:700;}
.muted {font-size:.68em;color:#667085;font-weight:600;}

.domain-grid {
  display:grid;
  grid-template-columns:repeat(5,minmax(0,1fr));
  gap:10px;
  margin:2px 0 10px;
}
.domain-row {
  background:#FFFFFF;
  border:1px solid #DDE1E7;
  box-shadow:0 3px 12px rgba(20,30,50,.04);
  border-radius:12px;
  padding:12px;
}
.domain-head {
  display:flex;
  min-height:38px;
  justify-content:space-between;
  gap:8px;
  font-size:12px;
  font-weight:700;
}
.domain-score {font-weight:900;}
.domain-score.good{color:var(--eami-good);}
.domain-score.mid{color:var(--eami-mid);}
.domain-score.bad{color:var(--eami-bad);}
.domain-score.neutral{color:#a1a3ad;}
.domain-track {
  height:6px;
  border-radius:999px;
  overflow:hidden;
  background:#E7EAF0;
  margin-top:10px;
}
.domain-fill {height:100%;border-radius:999px;background:#747886;}
.domain-fill.good{background:var(--eami-good);}
.domain-fill.mid{background:var(--eami-mid);}
.domain-fill.bad{background:var(--eami-bad);}
.domain-fill.neutral{background:#686b75;}
.domain-foot {
  color:#7A8291;
  font-size:9px;
  text-transform:uppercase;
  letter-spacing:.08em;
  margin-top:7px;
}

#action-bar {
  position:sticky;
  bottom:12px;
  z-index:25;
  padding:8px !important;
  border:1px solid #DDE1E7;
  background:rgba(255,255,255,.96);
  backdrop-filter:blur(14px);
  border-radius:15px;
  box-shadow:0 10px 28px rgba(31,41,55,.12);
}
#action-bar button {
  min-height:44px;
  border-radius:10px;
  font-weight:800;
}
#analyze-btn button {
  background:linear-gradient(90deg,#A50B5E,#7d0947);
  color:white;
  border:none;
}
#report-btn button {
  border:1px solid #5a5d68;
}

#key-findings {
  color:#344054 !important;
}

#key-findings h3 {
  color:#101828 !important;
}

#key-findings p,
#key-findings li {
  text-align: justify !important;
  text-align-last: left;
  text-justify: inter-word;
  line-height: 1.58;
  hyphens: none !important;
  -webkit-hyphens: none !important;
  word-break: normal;
  overflow-wrap: normal;
}

#key-findings ul {
  padding-left: 1.25rem;
}

#key-findings h3 {
  text-align: left !important;
  margin-top: 0.2rem;
  margin-bottom: 0.7rem;
}

#report-downloads {
  border:1px solid #DDE1E7;
  background:#FFFFFF;
  padding:14px;
  border-radius:14px;
  box-shadow:0 3px 14px rgba(20,30,50,.04);
}
#report-downloads button {
  min-height:38px !important;
  border-radius:9px !important;
  font-weight:800 !important;
}
#pdf-download button {
  background:linear-gradient(90deg,#A50B5E,#7d0947) !important;
  color:white !important;
  border:none !important;
}

.input-note {color:#8f919d;font-size:11px;}
.tight-accordion {margin-top:4px !important;}

@media (max-width:900px) {
  .domain-grid {grid-template-columns:1fr 1fr;}
  #hero {align-items:flex-start;flex-direction:column;}
}
@media (max-width:600px) {
  .domain-grid {grid-template-columns:1fr;}
}

/* =========================================================
   EAMI custom footer
   ========================================================= */

#eami-footer {
  margin-top:22px !important;
  margin-bottom:22px !important;
}

.eami-footer {
  padding:18px 20px 16px;

  background:#FFFFFF;
  border-top:1px solid #DDE1E7;

  text-align:center;
}

.eami-footer-brand {
  display:flex;
  justify-content:center;
  align-items:baseline;
  gap:7px;

  color:#344054;
}

.eami-footer-brand strong {
  color:#A50B5E;
  font-size:13px;
  font-weight:850;
  letter-spacing:.08em;
}

.eami-footer-brand span {
  font-size:11px;
  font-weight:650;
}

.eami-footer-meta {
  display:flex;
  flex-wrap:wrap;
  justify-content:center;
  align-items:center;
  gap:7px;

  margin-top:7px;

  color:#667085;
  font-size:9.5px;
  line-height:1.45;
}

.footer-dot {
  color:#C5CAD2;
}

.eami-footer-ruo {
  margin-top:7px;

  color:#98A2B3;
  font-size:9px;
  font-weight:650;
  letter-spacing:.025em;
}

@media (max-width:700px) {
  .eami-footer-meta {
    flex-direction:column;
    gap:3px;
  }

  .footer-dot {
    display:none;
  }

  .eami-footer-brand {
    flex-direction:column;
    gap:2px;
  }
}



/* =========================================================
   Molecular-domain information tooltips
   ========================================================= */

.domain-name-wrap {
  display:flex;
  align-items:flex-start;
  gap:6px;
  min-width:0;
}

.domain-help {
  position:relative;

  display:inline-flex;
  align-items:center;
  justify-content:center;

  flex:0 0 17px;
  width:17px;
  height:17px;

  margin-top:-1px;

  border:1px solid #C8CDD5;
  border-radius:50%;

  background:#FFFFFF;
  color:#667085;

  font-size:10px;
  line-height:1;
  font-weight:800;

  cursor:help;
  user-select:none;
}

.domain-help:hover,
.domain-help:focus {
  color:#A50B5E;
  border-color:#A50B5E;
  background:#FFF1F8;
  outline:none;
}

/* Tooltip box */
.domain-help::after {
  content:attr(data-tip);

  position:absolute;
  z-index:9999;

  left:0;
  bottom:calc(100% + 9px);

  width:275px;
  max-width:65vw;

  padding:10px 12px;

  background:#101828;
  color:#FFFFFF;

  border-radius:8px;

  box-shadow:0 8px 24px rgba(16,24,40,.20);

  font-size:10.5px;
  line-height:1.45;
  font-weight:500;
  text-align:left;

  opacity:0;
  visibility:hidden;
  pointer-events:none;

  transition:opacity .12s ease;
}

.domain-help::before {
  content:"";

  position:absolute;
  z-index:10000;

  left:4px;
  bottom:calc(100% + 4px);

  border-width:5px 5px 0 5px;
  border-style:solid;
  border-color:#101828 transparent transparent transparent;

  opacity:0;
  visibility:hidden;
}

.domain-help:hover::after,
.domain-help:hover::before,
.domain-help:focus::after,
.domain-help:focus::before {
  opacity:1;
  visibility:visible;
}



/* =========================================================
   Input data coverage
   ========================================================= */

#input-coverage {
  margin:16px 0 10px !important;
}

.input-coverage-card {
  padding:14px 16px;
  background:#FFFFFF;
  border:1px solid #DDE1E7;
  border-radius:12px;
  box-shadow:0 3px 12px rgba(16,24,40,.035);
}

.coverage-main {
  display:grid;
  grid-template-columns:180px 1fr;
  align-items:center;
  gap:18px;
}

.coverage-label {
  color:#667085;
  font-size:9px;
  font-weight:850;
  letter-spacing:.11em;
}

.coverage-value {
  color:#101828;
  font-size:20px;
  font-weight:800;
  margin-top:3px;
}

.coverage-value span {
  color:#667085;
  font-size:11px;
  font-weight:650;
  margin-left:5px;
}

.coverage-track {
  height:7px;
  overflow:hidden;
  border-radius:999px;
  background:#EAECF0;
}

.coverage-fill {
  height:100%;
  border-radius:999px;
  background:#A50B5E;
}

.coverage-breakdown {
  display:flex;
  flex-wrap:wrap;
  gap:18px;
  margin-top:10px;
  color:#667085;
  font-size:10.5px;
}

.coverage-breakdown strong {
  color:#344054;
}

.coverage-note {
  margin-top:7px;
  color:#98A2B3;
  font-size:9.5px;
}

.coverage-idle-text {
  color:#667085;
  font-size:11.5px;
  margin-top:4px;
}

.coverage-high {
  border-left:3px solid #2F855A;
}

.coverage-mid {
  border-left:3px solid #B7791F;
}

.coverage-low {
  border-left:3px solid #C2414B;
}

@media (max-width:700px) {
  .coverage-main {
    grid-template-columns:1fr;
  }
}



/* =========================================================
   Pre-analysis EAMI state
   ========================================================= */

#analysis-placeholder {
  margin-top:24px !important;
  margin-bottom:28px !important;
}

.analysis-empty-state {
  display:flex;
  align-items:center;
  gap:17px;

  padding:22px 24px;

  background:#FFFFFF;
  border:1px solid #DDE1E7;
  border-radius:14px;

  box-shadow:0 4px 16px rgba(16,24,40,.04);
}

.empty-state-mark {
  display:flex;
  align-items:center;
  justify-content:center;

  flex:0 0 54px;
  height:54px;

  background:#FFF1F8;
  border:1px solid rgba(165,11,94,.20);
  border-radius:12px;

  color:#A50B5E;
  font-size:11px;
  font-weight:850;
  letter-spacing:.10em;
}

.empty-state-title {
  color:#101828;
  font-size:16px;
  font-weight:750;
  line-height:1.3;
  margin-bottom:5px;
}

.empty-state-text {
  color:#667085;
  font-size:12.5px;
  line-height:1.5;
}

.empty-state-text strong {
  color:#A50B5E;
  font-weight:750;
}



/* =========================================================
   NEW SAMPLE control
   ========================================================= */

#clear-btn,
#clear-btn button {
  background:#FFFFFF !important;
  color:#475467 !important;
  border:1px solid #D0D5DD !important;
  border-radius:10px !important;
  font-weight:750 !important;
}

#clear-btn:hover,
#clear-btn button:hover {
  background:#FFF8FC !important;
  color:#A50B5E !important;
  border-color:#A50B5E !important;
}

#clear-btn:active,
#clear-btn button:active {
  background:#FFF1F8 !important;
}



/* =========================================================
   Reference sample loading status
   ========================================================= */

#load-status {
  margin-top:10px !important;
}

.sample-status {
  display:inline-flex;
  align-items:center;
  min-height:30px;
  padding:6px 11px;
  border-radius:8px;
  font-size:11px;
  font-weight:650;
  letter-spacing:.01em;
}

.sample-status.idle {
  background:#F2F4F7;
  border:1px solid #EAECF0;
  color:#667085;
}

.sample-status.ready {
  background:#ECFDF3;
  border:1px solid #ABEFC6;
  color:#067647;
}



/* =========================================================
   Material selector - hide radio circles
   ========================================================= */

#material-type input[type="radio"] {
  position:absolute !important;
  opacity:0 !important;
  pointer-events:none !important;
  width:0 !important;
  height:0 !important;
  margin:0 !important;
}

#material-type label {
  display:inline-flex !important;
  align-items:center !important;
  justify-content:center !important;
  gap:0 !important;
  min-width:110px !important;
  padding:10px 18px !important;
}

/* Selected material */
#material-type label:has(input:checked) {
  background:#FFF1F8 !important;
  color:#A50B5E !important;
  border-color:#A50B5E !important;
  box-shadow:0 0 0 2px rgba(165,11,94,.08) !important;
}

/* Unselected material */
#material-type label:not(:has(input:checked)) {
  background:#FFFFFF !important;
  color:#344054 !important;
  border-color:#D0D5DD !important;
}



/* Final vertical alignment of pathway numeric controls */
#immune-panel input[type="number"],
#scmc-panel input[type="number"] {
  transform:translateY(-5px) !important;
}

/* Move the reset control with the numeric field */
#immune-panel button[aria-label*="reset" i],
#scmc-panel button[aria-label*="reset" i],
#immune-panel button[title*="reset" i],
#scmc-panel button[title*="reset" i] {
  transform:translateY(-5px) !important;
}



/* =========================================================
   Pathway sliders - restore compact numeric controls
   ========================================================= */

#immune-panel input[type="number"],
#scmc-panel input[type="number"] {
  width:96px !important;
  min-width:96px !important;
  max-width:96px !important;

  height:34px !important;
  min-height:34px !important;

  padding:4px 28px 4px 10px !important;

  background:#FFFFFF !important;
  color:#101828 !important;
  border:1px solid #D0D5DD !important;
  border-radius:7px !important;

  font-size:13px !important;
  line-height:24px !important;
  font-weight:500 !important;

  opacity:1 !important;
  overflow:visible !important;
}

/* Keep increment/decrement arrows visible */
#immune-panel input[type="number"]::-webkit-inner-spin-button,
#immune-panel input[type="number"]::-webkit-outer-spin-button,
#scmc-panel input[type="number"]::-webkit-inner-spin-button,
#scmc-panel input[type="number"]::-webkit-outer-spin-button {
  -webkit-appearance:auto !important;
  appearance:auto !important;
  opacity:1 !important;
  display:block !important;
}

/* Give the slider component enough vertical room */
#immune-panel input[type="range"],
#scmc-panel input[type="range"] {
  margin-top:6px !important;
  margin-bottom:4px !important;
}

/* Pathway accordion titles */
#immune-panel,
#scmc-panel {
  background:#FFFFFF !important;
  border:1px solid #DDE1E7 !important;
  border-radius:10px !important;
}

#immune-panel summary,
#immune-panel summary span,
#immune-panel button,
#scmc-panel summary,
#scmc-panel summary span,
#scmc-panel button {
  color:#344054 !important;
  opacity:1 !important;
  font-weight:650 !important;
}



/* =========================================================
   Numeric input visibility
   ========================================================= */

.gradio-container input[type="number"] {
  background:#FFFFFF !important;
  color:#101828 !important;
  border:1px solid #D0D5DD !important;
  border-radius:8px !important;
  min-height:44px !important;
  padding-left:12px !important;
  padding-right:38px !important;
  opacity:1 !important;
  font-weight:500 !important;
}

/* Keep Chrome/Safari number steppers visible */
.gradio-container input[type="number"]::-webkit-inner-spin-button,
.gradio-container input[type="number"]::-webkit-outer-spin-button {
  -webkit-appearance:auto !important;
  appearance:auto !important;
  opacity:1 !important;
  display:block !important;
  cursor:pointer !important;
}

/* Clear focus state */
.gradio-container input[type="number"]:focus {
  border-color:#A50B5E !important;
  box-shadow:0 0 0 3px rgba(165,11,94,.08) !important;
  outline:none !important;
}



/* =========================================================
   Advanced accordion visibility
   ========================================================= */

#brca-accordion,
#mtvar-accordion,
#isoform-accordion,
#immune-ref-accordion {
  background:#FFFFFF !important;
  border:1px solid #DDE1E7 !important;
  border-radius:10px !important;
  opacity:1 !important;
}

#brca-accordion button,
#brca-accordion summary,
#brca-accordion span,
#mtvar-accordion button,
#mtvar-accordion summary,
#mtvar-accordion span,
#isoform-accordion button,
#isoform-accordion summary,
#isoform-accordion span,
#immune-ref-accordion button,
#immune-ref-accordion summary,
#immune-ref-accordion span {
  color:#344054 !important;
  opacity:1 !important;
  font-weight:650 !important;
}

#brca-accordion svg,
#mtvar-accordion svg,
#isoform-accordion svg,
#immune-ref-accordion svg {
  color:#667085 !important;
  opacity:1 !important;
}



/* =========================================================
   FINAL REPORT PANEL OVERRIDE
   ========================================================= */

/* Entire report area stays white */
#report-downloads,
#report-downloads > div,
#report-downloads > div > div,
#report-downloads .row,
#report-downloads .form,
#report-downloads .wrap,
#report-downloads .container {
  background:#FFFFFF !important;
  background-color:#FFFFFF !important;
  opacity:1 !important;
  filter:none !important;
}

/* Kill any inherited disabled-overlay effect */
#report-downloads *,
#report-downloads *:disabled,
#report-downloads [disabled],
#report-downloads [aria-disabled="true"] {
  opacity:1 !important;
  filter:none !important;
}

/* Summary */
#report-summary,
#report-summary > div,
#report-summary p {
  background:#FFFFFF !important;
  color:#344054 !important;
}

#report-summary p {
  font-size:13px !important;
  line-height:1.45 !important;
  margin:0 0 12px 0 !important;
}

#report-summary strong {
  color:#101828 !important;
}

/* HTML button before report generation */
#html-download button,
#html-download button:disabled {
  background:#FFFFFF !important;
  color:#667085 !important;
  border:1px solid #D0D5DD !important;
  opacity:1 !important;
  font-weight:750 !important;
  box-shadow:none !important;
}

/* PDF button before report generation:
   visually strong although functionally disabled */
#pdf-download button,
#pdf-download button:disabled {
  background:#A50B5E !important;
  color:#FFFFFF !important;
  border:1px solid #A50B5E !important;
  opacity:1 !important;
  font-weight:800 !important;
  box-shadow:0 3px 10px rgba(165,11,94,.14) !important;
}

/* Still communicate disabled state through cursor */
#html-download button:disabled,
#pdf-download button:disabled {
  cursor:not-allowed !important;
}

/* Once HTML becomes available */
#html-download button:not(:disabled) {
  color:#A50B5E !important;
  border-color:#A50B5E !important;
  background:#FFFFFF !important;
}

/* Once PDF becomes available */
#pdf-download button:not(:disabled) {
  background:#A50B5E !important;
  color:#FFFFFF !important;
  border-color:#A50B5E !important;
}

/* Directory note */
#report-downloads p,
#report-downloads span {
  color:#667085 !important;
}

#report-downloads code {
  background:#F2F4F7 !important;
  color:#344054 !important;
  border:1px solid #EAECF0 !important;
  border-radius:5px !important;
  padding:2px 5px !important;
}



/* =========================================================
   Report panel - pre-generation appearance
   ========================================================= */

#report-downloads {
  background:#FFFFFF !important;
  border:1px solid #DDE1E7 !important;
  border-radius:14px !important;
  padding:18px !important;
}

/* Remove inherited grey disabled-looking panel */
#report-downloads > div,
#report-downloads .form,
#report-downloads .row {
  background:#FFFFFF !important;
  opacity:1 !important;
}

/* Helper text */
#report-summary,
#report-summary p {
  background:transparent !important;
  color:#475467 !important;
  opacity:1 !important;
  font-weight:500 !important;
}

#report-summary p {
  margin-bottom:10px !important;
}

/* Disabled HTML button */
#html-download button:disabled {
  background:#F8FAFC !important;
  color:#667085 !important;
  border:1px solid #D0D5DD !important;
  opacity:1 !important;
  font-weight:700 !important;
}

/* Disabled PDF button */
#pdf-download button:disabled {
  background:#F7EAF1 !important;
  color:#A50B5E !important;
  border:1px solid #DFA9C6 !important;
  opacity:1 !important;
  font-weight:800 !important;
}

/* Enabled HTML button */
#html-download button:not(:disabled) {
  background:#FFFFFF !important;
  color:#A50B5E !important;
  border:1px solid #A50B5E !important;
  opacity:1 !important;
}

/* Enabled PDF button */
#pdf-download button:not(:disabled) {
  background:#A50B5E !important;
  color:#FFFFFF !important;
  border:1px solid #A50B5E !important;
  opacity:1 !important;
}

/* Report directory note */
#report-downloads p,
#report-downloads span {
  color:#475467 !important;
  opacity:1 !important;
}

#report-downloads code {
  background:#F2F4F7 !important;
  color:#344054 !important;
  border:1px solid #EAECF0 !important;
}



/* =========================================================
   EAMI stable application shell
   ========================================================= */

/* Outer browser area */
html,
body {
  margin:0 !important;
  padding:0 !important;
  background:#E4E7EC !important;
  overflow-x:hidden !important;
}

/* Gradio outer application element */
gradio-app {
  background:#E4E7EC !important;
}

/* Keep EAMI at one stable width before and after analysis */
.gradio-container {
  width:calc(100% - 48px) !important;
  max-width:1280px !important;
  min-width:0 !important;
  margin:0 auto !important;
  padding-left:34px !important;
  padding-right:34px !important;
  box-sizing:border-box !important;

  background:#F7F8FA !important;
  min-height:100vh !important;

  box-shadow:
    0 0 0 1px rgba(16,24,40,.03),
    0 10px 40px rgba(16,24,40,.05) !important;
}

/* Prevent dynamically loaded content from widening the application */
.gradio-container *,
.gradio-container *::before,
.gradio-container *::after {
  box-sizing:border-box !important;
}

.gradio-container .row,
.gradio-container .column,
.gradio-container .form {
  min-width:0 !important;
  max-width:100% !important;
}

/* Plots, tables and dynamic outputs stay inside the fixed shell */
.gradio-container img,
.gradio-container canvas,
.gradio-container svg {
  max-width:100% !important;
}

#domain-table,
#cohort-table,
#key-findings,
#report-downloads,
#audit-trail {
  min-width:0 !important;
  max-width:100% !important;
}

/* Avoid horizontal layout shift from large generated elements */
.gradio-container {
  overflow-x:hidden !important;
}

/* Responsive behaviour */
@media (max-width:900px) {
  .gradio-container {
    width:calc(100% - 24px) !important;
    padding-left:20px !important;
    padding-right:20px !important;
  }
}

@media (max-width:600px) {
  .gradio-container {
    width:100% !important;
    padding-left:14px !important;
    padding-right:14px !important;
  }

  html,
  body,
  gradio-app {
    background:#F7F8FA !important;
  }
}



/* =========================================================
   Gradio Dataframe definitive light theme
   ========================================================= */

#domain-table,
#cohort-table {

  /* Gradio Dataframe-specific variables */
  --gr-df-table-bg-even:#FFFFFF !important;
  --gr-df-table-bg-odd:#F8FAFC !important;
  --gr-df-table-text:#344054 !important;
  --gr-df-table-border:#D0D5DD !important;
  --gr-df-accent:#A50B5E !important;
  --gr-df-accent-soft:#FFF1F8 !important;

  /* Standard Gradio table variables */
  --table-even-background-fill:#FFFFFF !important;
  --table-even-background-fill-dark:#FFFFFF !important;

  --table-odd-background-fill:#F8FAFC !important;
  --table-odd-background-fill-dark:#F8FAFC !important;

  --table-text-color:#344054 !important;
  --table-text-color-dark:#344054 !important;

  --table-border-color:#D0D5DD !important;
  --table-border-color-dark:#D0D5DD !important;

  --table-row-focus:#FFF1F8 !important;
  --table-row-focus-dark:#FFF1F8 !important;

  background:#FFFFFF !important;
  color:#344054 !important;
}

/* Extra safeguard against inherited dark mode */
#domain-table table,
#cohort-table table,
#domain-table tbody,
#cohort-table tbody {
  background:#FFFFFF !important;
  color:#344054 !important;
}

#domain-table tbody tr:nth-child(odd),
#cohort-table tbody tr:nth-child(odd) {
  background:#FFFFFF !important;
}

#domain-table tbody tr:nth-child(even),
#cohort-table tbody tr:nth-child(even) {
  background:#F8FAFC !important;
}

#domain-table tbody td,
#cohort-table tbody td {
  color:#344054 !important;
}

/* Cleaner table typography */
#domain-table,
#cohort-table {
  font-size:12px !important;
}



/* =========================================================
   Audit trail tables - light academic appearance
   ========================================================= */

#domain-table,
#cohort-table {
  background:#FFFFFF !important;
  color:#344054 !important;
  border:1px solid #DDE1E7 !important;
  border-radius:9px !important;
  overflow:hidden !important;
}

/* Table container */
#domain-table .table-wrap,
#cohort-table .table-wrap,
#domain-table .wrap,
#cohort-table .wrap {
  background:#FFFFFF !important;
}

/* Main table */
#domain-table table,
#cohort-table table {
  background:#FFFFFF !important;
  color:#344054 !important;
  border-collapse:collapse !important;
}

/* Header */
#domain-table thead,
#cohort-table thead,
#domain-table th,
#cohort-table th {
  background:#F2F4F7 !important;
  color:#344054 !important;
  font-weight:700 !important;
  border-color:#D0D5DD !important;
}

/* Cells */
#domain-table td,
#cohort-table td {
  background:#FFFFFF !important;
  color:#344054 !important;
  border-color:#EAECF0 !important;
}

/* Alternate rows */
#domain-table tbody tr:nth-child(even) td,
#cohort-table tbody tr:nth-child(even) td {
  background:#F9FAFB !important;
}

/* Hover */
#domain-table tbody tr:hover td,
#cohort-table tbody tr:hover td {
  background:#FFF8FC !important;
}

/* Any input-like cell rendering */
#domain-table input,
#cohort-table input,
#domain-table textarea,
#cohort-table textarea {
  background:#FFFFFF !important;
  color:#344054 !important;
}

/* Column text */
#domain-table *,
#cohort-table * {
  text-shadow:none !important;
}



/* =========================================================
   EAMI lower interface refinement
   ========================================================= */

/* Audit trail accordion */
#audit-trail {
  background:#FFFFFF !important;
  border:1px solid #DDE1E7 !important;
  border-radius:10px !important;
  opacity:1 !important;
}

#audit-trail button,
#audit-trail summary,
#audit-trail span,
#audit-trail p {
  color:#344054 !important;
  opacity:1 !important;
}

#audit-trail > button,
#audit-trail > summary {
  font-weight:650 !important;
  background:#FFFFFF !important;
}

#audit-trail svg {
  color:#667085 !important;
  opacity:1 !important;
}

/* Report block */
#report-downloads {
  background:#FFFFFF !important;
  border:1px solid #DDE1E7 !important;
  border-radius:14px !important;
  padding:18px !important;
  box-shadow:0 4px 16px rgba(16,24,40,.04) !important;
}

#report-summary {
  background:transparent !important;
  border:none !important;
  color:#344054 !important;
  padding:0 0 10px 0 !important;
}

#report-summary p {
  color:#344054 !important;
  margin:0 !important;
}

/* Enabled HTML report */
#html-download button:not(:disabled) {
  background:#FFFFFF !important;
  color:#A50B5E !important;
  border:1px solid #A50B5E !important;
  opacity:1 !important;
}

/* Enabled PDF report */
#pdf-download button:not(:disabled) {
  background:#A50B5E !important;
  color:#FFFFFF !important;
  border:1px solid #A50B5E !important;
  opacity:1 !important;
}

/* Disabled report buttons: visibly inactive, but readable */
#html-download button:disabled,
#pdf-download button:disabled {
  background:#F2F4F7 !important;
  color:#98A2B3 !important;
  border:1px solid #D0D5DD !important;
  opacity:1 !important;
  cursor:not-allowed !important;
}

/* Report location text */
#report-downloads p,
#report-downloads span {
  color:#475467 !important;
}

#report-downloads code {
  background:#F2F4F7 !important;
  color:#344054 !important;
  border:1px solid #EAECF0 !important;
  border-radius:5px !important;
  padding:2px 5px !important;
}

/* Research-use notice */
#ruo-footer {
  margin-top:18px !important;
  padding:13px 16px !important;
  background:#FFF8FC !important;
  border:1px solid rgba(165,11,94,.22) !important;
  border-left:4px solid #A50B5E !important;
  border-radius:9px !important;
}

#ruo-footer p {
  margin:0 !important;
  color:#344054 !important;
  font-size:13px !important;
}

#ruo-footer strong {
  color:#A50B5E !important;
}

/* Hide Gradio's own footer for a cleaner research-demo interface */
.gradio-container footer {
  display:none !important;
}



/* =========================================================
   Accordion header visibility
   ========================================================= */

#reference-loader,
#report-metadata {
  opacity:1 !important;
}

#reference-loader button,
#reference-loader summary,
#reference-loader span,
#reference-loader p,
#report-metadata button,
#report-metadata summary,
#report-metadata span,
#report-metadata p {
  color:#344054 !important;
  opacity:1 !important;
}

#reference-loader > button,
#report-metadata > button {
  background:#FFFFFF !important;
  color:#344054 !important;
  font-weight:650 !important;
}

#reference-loader svg,
#report-metadata svg {
  color:#667085 !important;
  opacity:1 !important;
}

/* Slight visual separation for closed accordions */
#reference-loader,
#report-metadata {
  background:#FFFFFF !important;
  border:1px solid #DDE1E7 !important;
  border-radius:10px !important;
}



/* =========================================================
   Material selector + demo reference controls
   ========================================================= */

#material-type {
  background:#FFFFFF !important;
  border:1px solid #DDE1E7 !important;
  border-radius:12px !important;
  padding:14px 16px !important;
}

#material-type label {
  background:#FFFFFF !important;
  color:#344054 !important;
  border:1px solid #D0D5DD !important;
  border-radius:9px !important;
  padding:8px 14px !important;
  font-weight:600 !important;
  transition:all .15s ease;
}

#material-type label:hover {
  border-color:#A50B5E !important;
  background:#FFF8FC !important;
}

#material-type label:has(input:checked) {
  background:#FFF1F8 !important;
  color:#A50B5E !important;
  border-color:#A50B5E !important;
  box-shadow:0 0 0 2px rgba(165,11,94,.08) !important;
}

#material-type input[type="radio"] {
  accent-color:#A50B5E !important;
}

/* Reference panel */
#reference-loader {
  background:#FFFFFF !important;
  border:1px solid #DDE1E7 !important;
  border-radius:12px !important;
}

#reference-loader summary,
#reference-loader summary span {
  color:#344054 !important;
  font-weight:600 !important;
}

#reference-loader label {
  color:#344054 !important;
}

/* Load reference button */
#load-reference,
#load-reference button {
  background:#FFFFFF !important;
  color:#A50B5E !important;
  border:1px solid #A50B5E !important;
  border-radius:9px !important;
  font-weight:700 !important;
}

#load-reference:hover,
#load-reference button:hover {
  background:#FFF1F8 !important;
  color:#87094D !important;
  border-color:#87094D !important;
}



.eami-kicker {
  color:#A50B5E;
  font-size:10px;
  font-weight:800;
  letter-spacing:.15em;
  margin-bottom:9px;
}

#hero {
  align-items:center !important;
  padding:20px 2px 8px;
}

#hero h1 {
  color:#101828 !important;
  font-size:52px !important;
  font-weight:800;
}

#hero .sub {
  color:#475467 !important;
  font-size:17px !important;
  font-weight:600;
}

.hero-description {
  color:#667085;
  font-size:13px;
  margin-top:9px;
  max-width:670px;
  line-height:1.5;
}

.demo-badge {
  min-width:190px;
  padding:11px 15px;
  border:1px solid rgba(165,11,94,.28);
  border-radius:12px;
  background:#FFF8FC;
  text-align:left;
}

.demo-badge strong {
  display:block;
  color:#A50B5E;
  font-size:11px;
  letter-spacing:.08em;
}

.demo-badge span {
  display:block;
  margin-top:3px;
  color:#667085;
  font-size:11px;
}

.eami-workflow {
  display:grid;
  grid-template-columns:1fr auto 1.6fr auto 1.15fr;
  gap:14px;
  align-items:center;
  padding:15px 17px;
  margin:8px 0 24px;
  background:#FFFFFF;
  border:1px solid #E1E5EA;
  border-radius:14px;
  box-shadow:0 4px 16px rgba(16,24,40,.04);
}

.workflow-step {
  display:flex;
  gap:10px;
  align-items:flex-start;
}

.workflow-number {
  color:#A50B5E;
  font-size:10px;
  font-weight:800;
  padding-top:2px;
}

.workflow-step strong {
  display:block;
  color:#101828;
  font-size:10px;
  letter-spacing:.07em;
}

.workflow-step span {
  display:block;
  color:#667085;
  font-size:11px;
  line-height:1.35;
  margin-top:3px;
}

.workflow-arrow {
  color:#98A2B3;
  font-size:18px;
}

@media (max-width:900px) {
  .eami-workflow {
    grid-template-columns:1fr;
  }

  .workflow-arrow {
    transform:rotate(90deg);
    text-align:center;
  }

  .demo-badge {
    min-width:0;
  }
}


/* =========================================================
   EAMI v0.8.2 — academic light interface
   ========================================================= */

:root,
.dark {
  --body-background-fill:#F7F8FA;
  --body-background-fill-dark:#F7F8FA;
  --body-text-color:#101828;
  --body-text-color-subdued:#667085;

  --background-fill-primary:#FFFFFF;
  --background-fill-secondary:#F2F4F7;

  --block-background-fill:#FFFFFF;
  --block-border-color:#DDE1E7;
  --block-label-text-color:#344054;
  --block-title-text-color:#101828;

  --input-background-fill:#FFFFFF;
  --input-border-color:#D0D5DD;
  --input-border-color-focus:#A50B5E;

  --button-primary-background-fill:#A50B5E;
  --button-primary-background-fill-hover:#87094D;
  --button-primary-text-color:#FFFFFF;

  --button-secondary-background-fill:#FFFFFF;
  --button-secondary-background-fill-hover:#F2F4F7;
  --button-secondary-text-color:#344054;

  --border-color-primary:#D0D5DD;
  --border-color-accent:#A50B5E;
}

/* Core text */
.gradio-container,
.gradio-container p,
.gradio-container li,
.gradio-container label {
  color:#344054;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4 {
  color:#101828;
}

/* EAMI result cards */
.ui-card {
  color:#101828 !important;
}

.ui-value {
  color:#101828 !important;
}

.ui-eyebrow {
  color:#667085 !important;
}

.ui-sub {
  color:#667085 !important;
}

/* Domain cards */
.domain-row {
  color:#101828 !important;
}

.domain-head {
  color:#101828 !important;
}

.domain-foot {
  color:#667085 !important;
}

.domain-track {
  background:#EAECF0 !important;
}

/* Input controls */
.gradio-container input,
.gradio-container textarea {
  background:#FFFFFF !important;
  color:#101828 !important;
  border-color:#D0D5DD !important;
}

/* Tabs */
.gradio-container button[role="tab"] {
  color:#667085 !important;
  background:transparent !important;
}

.gradio-container button[role="tab"][aria-selected="true"] {
  color:#A50B5E !important;
  background:#FFFFFF !important;
  border-color:#A50B5E !important;
}

/* Accordions and panels */
.gradio-container details {
  background:#FFFFFF !important;
  color:#344054 !important;
  border-color:#DDE1E7 !important;
}

.gradio-container summary {
  background:#FFFFFF !important;
  color:#344054 !important;
}

/* Main actions */
#analyze-btn,
#analyze-btn button {
  background:#A50B5E !important;
  color:#FFFFFF !important;
  border:none !important;
}

#analyze-btn:hover,
#analyze-btn button:hover {
  background:#87094D !important;
}

#report-btn,
#report-btn button {
  background:#FFFFFF !important;
  color:#344054 !important;
  border:1px solid #D0D5DD !important;
}

/* Interpretation text */
#key-findings {
  color:#344054 !important;
}

#key-findings p,
#key-findings li {
  color:#344054 !important;
}

#key-findings h3 {
  color:#101828 !important;
}

/* Section headings */
.section-title {
  color:#667085 !important;
}

/* Scroll / visual polish */
.gradio-container {
  background:#F7F8FA !important;
}



/* =========================================================
   FINAL EAMI WHITE RESULTS AREA
   ========================================================= */

/* Results wrapper created for the pre-analysis state */
#results-area {
  background:#FFFFFF !important;
  background-color:#FFFFFF !important;
  border:none !important;
  box-shadow:none !important;
}

/* Gradio may add nested wrappers inside Group */
#results-area > div,
#results-area > div > div {
  background:#FFFFFF !important;
  background-color:#FFFFFF !important;
}

/* Main result sections stay white */
#results-area .gradio-group,
#results-area .group,
#results-area .form,
#results-area .row,
#results-area .column {
  background-color:transparent !important;
}

/* Keep the application itself white */
.gradio-container {
  background:#FFFFFF !important;
  background-color:#FFFFFF !important;
}

/* Keep only the OUTER side margins light grey */
html,
body,
gradio-app {
  background:#E4E7EC !important;
}

/* Cards remain white */
.ui-card,
.domain-row {
  background:#FFFFFF !important;
}




/* =========================================================
   EAMI SOFT PANEL BALANCE
   White results area + softly tinted control panels
   ========================================================= */

:root{
  --eami-soft-panel:#F6F7FA;
  --eami-soft-panel-2:#F9FAFC;
  --eami-soft-border:#D9DFE8;
}

/* Whole app stays white */
.gradio-container{
  background:#FFFFFF !important;
  background-color:#FFFFFF !important;
}

/* Outer browser-side margins stay light gray */
html, body, gradio-app{
  background:#E4E7EC !important;
}

/* Default panels / groups get soft light background */
.gradio-group,
.block,
.panel {
  background:var(--eami-soft-panel) !important;
  background-color:var(--eami-soft-panel) !important;
  border:1px solid var(--eami-soft-border) !important;
  box-shadow:none !important;
}

/* Inner cards remain white for contrast */
.ui-card,
.domain-row,
.metric-card,
.summary-card,
.evidence-card,
.report-card {
  background:#FFFFFF !important;
  background-color:#FFFFFF !important;
}

/* Accordions and collapsible headers */
.gradio-accordion,
.gradio-accordion > div,
.gradio-accordion summary {
  background:var(--eami-soft-panel-2) !important;
  background-color:var(--eami-soft-panel-2) !important;
}

/* INPUT / control area can stay softly tinted */
#hero,
#input-workspace,
#controls-area,
#top-controls {
  background:var(--eami-soft-panel) !important;
  background-color:var(--eami-soft-panel) !important;
}

/* RESULTS AREA must stay white */
#results-area,
#results-area > div,
#results-area > div > div,
#results-area .gradio-group,
#results-area .block,
#results-area .panel,
#results-area .gradio-accordion,
#results-area .gradio-accordion > div,
#results-area .gradio-accordion summary {
  background:#FFFFFF !important;
  background-color:#FFFFFF !important;
  border-color:#DCE2EA !important;
}

/* Keep white report area */
#results-area .ui-card,
#results-area .domain-row,
#results-area .metric-card,
#results-area .summary-card,
#results-area .evidence-card,
#results-area .report-card {
  background:#FFFFFF !important;
  background-color:#FFFFFF !important;
}




/* =========================================================
   EAMI APPLE-LIKE OVERRIDE
   Clean white UI + selective highlight cards
   ========================================================= */

:root{
  --apple-bg:#F5F5F7;
  --apple-panel:#FFFFFF;
  --apple-border:#E5E7EB;
  --apple-text:#1F2937;
  --apple-muted:#667085;
  --apple-shadow:0 10px 28px rgba(15,23,42,0.06);

  --eami-accent:#A50B5E;
  --eami-accent-soft:#FAEEF5;
  --eami-blue-soft:#EFF6FF;
  --eami-green-soft:#ECFDF3;
  --eami-amber-soft:#FFF7ED;
}

/* Page background */
html, body, gradio-app{
  background:var(--apple-bg) !important;
}

/* Main app canvas */
.gradio-container{
  background:transparent !important;
  color:var(--apple-text) !important;
}

/* STOP the old gray slab effect */
.gradio-group,
.block,
.panel{
  background:#FFFFFF !important;
  background-color:#FFFFFF !important;
  border:1px solid var(--apple-border) !important;
  border-radius:18px !important;
  box-shadow:var(--apple-shadow) !important;
}

/* Main structural wrappers should stay clean */
#hero,
#results-area,
#controls-area,
#input-workspace,
#top-controls{
  background:transparent !important;
  border:none !important;
  box-shadow:none !important;
}

/* Common cards */
.ui-card,
.metric-card,
.summary-card,
.evidence-card,
.report-card,
.domain-row{
  background:#FFFFFF !important;
  background-color:#FFFFFF !important;
  border:1px solid var(--apple-border) !important;
  border-radius:18px !important;
  box-shadow:var(--apple-shadow) !important;
}

/* Section shells */
.section-shell{
  background:#FFFFFF !important;
  border:1px solid var(--apple-border) !important;
  border-radius:20px !important;
  box-shadow:var(--apple-shadow) !important;
}

/* Results blocks must be white */
#results-area .gradio-group,
#results-area .block,
#results-area .panel,
#results-area .gradio-accordion,
#results-area .gradio-accordion > div,
#results-area .gradio-accordion summary{
  background:#FFFFFF !important;
  background-color:#FFFFFF !important;
  border-color:var(--apple-border) !important;
  box-shadow:none !important;
}

/* Summary cards – Apple-like emphasis */
.summary-card:nth-of-type(1){
  background:var(--eami-accent-soft) !important;
  border:1px solid rgba(165,11,94,0.30) !important;
}
.summary-card:nth-of-type(2){
  background:#FFFFFF !important;
  border:1px solid #D7DCE5 !important;
}
.summary-card:nth-of-type(3){
  background:var(--eami-blue-soft) !important;
  border:1px solid rgba(59,130,246,0.20) !important;
}
.summary-card:nth-of-type(4){
  background:var(--eami-green-soft) !important;
  border:1px solid rgba(34,197,94,0.24) !important;
}

/* Domain cards stay mostly white but crisp */
.metric-card{
  background:#FFFFFF !important;
}

/* Inputs */
input, textarea, select{
  background:#FFFFFF !important;
  color:var(--apple-text) !important;
  border:1px solid #D0D5DD !important;
  border-radius:14px !important;
  box-shadow:none !important;
}

input::placeholder,
textarea::placeholder{
  color:#98A2B3 !important;
}

/* Number inputs */
input[type="number"]{
  color:var(--apple-text) !important;
}

/* Buttons */
button{
  border-radius:14px !important;
  font-weight:700 !important;
  box-shadow:none !important;
}

button.primary,
.primary{
  background:var(--eami-accent) !important;
  color:#FFFFFF !important;
  border:1px solid var(--eami-accent) !important;
}

button.secondary,
.secondary{
  background:#FFFFFF !important;
  color:#344054 !important;
  border:1px solid #D0D5DD !important;
}

/* Tabs */
[role="tablist"]{
  background:transparent !important;
  border:none !important;
}

button[role="tab"]{
  background:transparent !important;
  color:#667085 !important;
  border-radius:12px !important;
  border:1px solid transparent !important;
}

button[role="tab"][aria-selected="true"]{
  background:#FFFFFF !important;
  color:var(--eami-accent) !important;
  border:1px solid var(--apple-border) !important;
  box-shadow:0 3px 12px rgba(15,23,42,0.05) !important;
}

/* Accordion headers */
.gradio-accordion summary{
  background:#FFFFFF !important;
  color:#344054 !important;
  border-radius:14px !important;
}

/* Success chip */
.status-badge,
.success-chip{
  background:#ECFDF3 !important;
  color:#067647 !important;
  border:1px solid #ABEFC6 !important;
  border-radius:999px !important;
}

/* Small labels / helper text */
label,
.gr-label,
.help-text,
small{
  color:var(--apple-muted) !important;
}

/* Preserve white content zones */
#results-area .summary-card,
#results-area .metric-card,
#results-area .evidence-card,
#results-area .report-card{
  background-color:#FFFFFF !important;
}

/* Slight tint only for top control zone cards */
#hero .section-shell,
#controls-area .section-shell,
#input-workspace > .section-shell{
  background:#FBFBFC !important;
}

/* Make interpretation panel clean */
#results-area .interpretation-card,
#results-area .keyfindings-card{
  background:#FFFFFF !important;
  border:1px solid var(--apple-border) !important;
}

/* Remove any previous broad gray soft-panel look */
[class*="soft-panel"],
[class*="soft-panel-2"]{
  background:transparent !important;
  background-color:transparent !important;
}



/* =========================================================
   FINAL SEMANTIC PROFILE COLORS
   CLEAN = green · INTERMEDIATE = amber · DIRTY = red
   ========================================================= */

/* CLEAN-like */
.ui-card.good {
  background:linear-gradient(180deg,#F0FDF4 0%,#ECFDF3 100%) !important;
  border:1px solid #86D9A7 !important;
  box-shadow:0 8px 22px rgba(22,101,52,.07) !important;
}

.ui-card.good .ui-eyebrow {
  color:#297A4A !important;
}

.ui-card.good .ui-value {
  color:#166534 !important;
}

.ui-card.good .ui-sub {
  color:#43835D !important;
}


/* INTERMEDIATE */
.ui-card.mid {
  background:linear-gradient(180deg,#FFFBEB 0%,#FFF7E6 100%) !important;
  border:1px solid #E8C36A !important;
  box-shadow:0 8px 22px rgba(180,83,9,.07) !important;
}

.ui-card.mid .ui-eyebrow {
  color:#A16207 !important;
}

.ui-card.mid .ui-value {
  color:#92400E !important;
}

.ui-card.mid .ui-sub {
  color:#A16207 !important;
}


/* DIRTY-like */
.ui-card.bad {
  background:linear-gradient(180deg,#FFF1F2 0%,#FEF2F2 100%) !important;
  border:1px solid #E69AA5 !important;
  box-shadow:0 8px 22px rgba(185,28,28,.07) !important;
}

.ui-card.bad .ui-eyebrow {
  color:#B42335 !important;
}

.ui-card.bad .ui-value {
  color:#9F1239 !important;
}

.ui-card.bad .ui-sub {
  color:#B54757 !important;
}


/* Oocyte profile remains visually separate */
.ui-card.oocyte {
  background:linear-gradient(180deg,#EFF6FF 0%,#F5F8FF 100%) !important;
  border:1px solid #9CB7DF !important;
}

.ui-card.oocyte .ui-value {
  color:#315B92 !important;
}




/* =========================================================
   COMPACT MOLECULAR INTERPRETATION
   ========================================================= */

#key-findings {
  padding:14px 18px !important;
  background:#FFFFFF !important;
  border:none !important;
  box-shadow:none !important;
}

/* Headings */
#key-findings h3 {
  margin:0 0 9px 0 !important;
  color:#101828 !important;
  font-size:15px !important;
  line-height:1.25 !important;
  font-weight:750 !important;
}

/* Findings list */
#key-findings ul {
  list-style:none !important;
  margin:0 0 16px 0 !important;
  padding:0 !important;
}

#key-findings li {
  position:relative;
  margin:0 0 7px 0 !important;
  padding:0 0 0 15px !important;

  color:#475467 !important;
  font-size:12.5px !important;
  line-height:1.42 !important;
}

/* Small Jazzberry bullet */
#key-findings li::before {
  content:"";
  position:absolute;
  left:1px;
  top:.55em;

  width:5px;
  height:5px;

  background:#A50B5E;
  border-radius:50%;
}

/* Bold labels */
#key-findings strong {
  color:#1D2939 !important;
  font-weight:700 !important;
}

/* Interpretation paragraph */
#key-findings p {
  margin:0 !important;
  padding:10px 12px !important;

  background:#FFF8FC !important;
  border-left:3px solid #A50B5E !important;
  border-radius:0 9px 9px 0 !important;

  color:#475467 !important;
  font-size:12.5px !important;
  line-height:1.48 !important;
}




/* =========================================================
   FINAL FIX — REMOVE GRAY CORNER ARTIFACTS
   ========================================================= */

/* Gradio parent wrappers must not paint gray behind rounded cards */
.gradio-container .gradio-group,
.gradio-container .group,
.gradio-container .panel,
.gradio-container .form {
  background:transparent !important;
  background-color:transparent !important;
  box-shadow:none !important;
}

/* Remove unwanted shadow/background from immediate wrappers */
.gradio-container .gradio-group > div,
.gradio-container .group > div,
.gradio-container .panel > div,
.gradio-container .form > div {
  box-shadow:none !important;
}

/* Main visible EAMI cards */
.ui-card,
.domain-row,
.input-coverage-card,
.analysis-empty-state,
.eami-footer,
#report-downloads,
#material-type,
#results-area .ui-card,
#results-area .domain-row {
  background-clip:padding-box !important;
  border-radius:18px !important;
  overflow:hidden !important;
}

/* White cards stay truly white to the rounded edge */
.ui-card,
.domain-row,
#report-downloads {
  background:#FFFFFF !important;
}

/* Preserve semantic colors */
.ui-card.good {
  background:linear-gradient(180deg,#F0FDF4 0%,#ECFDF3 100%) !important;
}

.ui-card.mid {
  background:linear-gradient(180deg,#FFFBEB 0%,#FFF7E6 100%) !important;
}

.ui-card.bad {
  background:linear-gradient(180deg,#FFF1F2 0%,#FEF2F2 100%) !important;
}

.ui-card.oocyte {
  background:linear-gradient(180deg,#EFF6FF 0%,#F5F8FF 100%) !important;
}

/* Prevent pseudo-elements from producing square/gray corners */
.gradio-container .gradio-group::before,
.gradio-container .gradio-group::after,
.gradio-container .group::before,
.gradio-container .group::after,
.gradio-container .panel::before,
.gradio-container .panel::after {
  background:transparent !important;
  box-shadow:none !important;
}

/* Keep the global Apple-like page background */
html,
body,
gradio-app {
  background:#F5F5F7 !important;
}




/* =========================================================
   REMOVE EXTRA INNER LINES / DOUBLE BORDERS
   ========================================================= */

:root{
  --eami-line:#D8DEE8;
  --eami-line-soft:rgba(15,23,42,0.06);
}

/* 1) structural Gradio wrappers should NOT draw their own borders */
.gradio-container .gradio-group,
.gradio-container .gr-group,
.gradio-container .form,
.gradio-container .panel,
.gradio-container fieldset {
  border:none !important;
  outline:none !important;
  box-shadow:none !important;
  background:transparent !important;
}

/* 2) direct inner wrapper divs also should not paint seams */
.gradio-container .gradio-group > div,
.gradio-container .gr-group > div,
.gradio-container .form > div,
.gradio-container .panel > div,
.gradio-container fieldset > div {
  border:none !important;
  outline:none !important;
  box-shadow:none !important;
  background:transparent !important;
}

/* 3) keep ONLY the visible card border */
.ui-card,
.domain-row,
.input-coverage-card,
.analysis-empty-state,
.eami-footer,
#report-downloads,
#material-type {
  border:1px solid var(--eami-line) !important;
  box-shadow:none !important;
  outline:none !important;
  background-clip:padding-box !important;
  overflow:hidden !important;
}

/* 4) remove those tiny inner separator lines inside cards */
.ui-card > div,
.domain-row > div,
.input-coverage-card > div,
.analysis-empty-state > div,
.eami-footer > div,
#report-downloads > div,
#material-type > div {
  border:none !important;
  outline:none !important;
  box-shadow:none !important;
  background:transparent !important;
}

/* 5) section wrappers should be clean */
#results-area,
#results-area > div,
#results-area .gradio-group,
#results-area .gr-group,
#results-area .form,
#results-area .panel {
  border:none !important;
  outline:none !important;
  box-shadow:none !important;
  background:transparent !important;
}

/* 6) preserve semantic card colors */
.ui-card.good{
  background:linear-gradient(180deg,#F0FDF4 0%,#ECFDF3 100%) !important;
}
.ui-card.mid{
  background:linear-gradient(180deg,#FFFBEB 0%,#FFF7E6 100%) !important;
}
.ui-card.bad{
  background:linear-gradient(180deg,#FFF1F2 0%,#FEF2F2 100%) !important;
}
.ui-card.oocyte{
  background:linear-gradient(180deg,#EFF6FF 0%,#F5F8FF 100%) !important;
}

/* 7) keep normal inputs/buttons looking normal */
input, textarea, select, button {
  box-shadow:none !important;
}

/* 8) no unwanted pseudo-element borders */
.gradio-container *::before,
.gradio-container *::after{
  box-shadow:none !important;
}




/* =========================================================
   EXECUTIVE SUMMARY — FINAL WRAPPER CLEANUP
   Only affects the four top molecular-summary cards
   ========================================================= */

#exec-summary-row {
  background:transparent !important;
  border:none !important;
  outline:none !important;
  box-shadow:none !important;
  padding:0 !important;
}

/* Remove the redundant Gradio shells around each HTML card */
#exec-summary-row > div,
#exec-summary-row > div > div,
#exec-summary-row .block,
#exec-summary-row .gradio-html,
#exec-summary-row .html-container {
  background:transparent !important;
  border:none !important;
  outline:none !important;
  box-shadow:none !important;
}

/* Remove extra padding that creates the visible second contour */
#exec-summary-row > div {
  padding:0 !important;
}

/* Keep ONLY the actual EAMI card */
#exec-summary-row .ui-card {
  border-radius:28px !important;
  overflow:hidden !important;
  box-shadow:none !important;
}

/* Neutral card */
#exec-summary-row .ui-card.neutral {
  background:#FFFFFF !important;
  border:1px solid #D8DEE8 !important;
}

/* CLEAN */
#exec-summary-row .ui-card.good {
  background:linear-gradient(
    180deg,
    #F0FDF4 0%,
    #ECFDF3 100%
  ) !important;
  border:1px solid #86D9A7 !important;
}

/* INTERMEDIATE */
#exec-summary-row .ui-card.mid {
  background:linear-gradient(
    180deg,
    #FFFBEB 0%,
    #FFF7E6 100%
  ) !important;
  border:1px solid #E8C36A !important;
}

/* DIRTY */
#exec-summary-row .ui-card.bad {
  background:linear-gradient(
    180deg,
    #FFF1F2 0%,
    #FEF2F2 100%
  ) !important;
  border:1px solid #E69AA5 !important;
}

/* Oocyte */
#exec-summary-row .ui-card.oocyte {
  background:linear-gradient(
    180deg,
    #EFF6FF 0%,
    #F5F8FF 100%
  ) !important;
  border:1px solid #9CB7DF !important;
}

/* Nothing inside the actual card should create another contour */
#exec-summary-row .ui-card > div {
  border:none !important;
  outline:none !important;
  box-shadow:none !important;
}




/* =========================================================
   FINAL TOOLTIP VISIBILITY FIX
   ========================================================= */

/* Tooltips must be allowed to leave the domain card */
.domain-grid,
.domain-row,
.domain-row > div,
.domain-head,
.domain-name-wrap {
  overflow:visible !important;
}

/* Keep the rounded card itself visually clean */
.domain-row {
  position:relative !important;
  z-index:1;
  border-radius:18px !important;
}

/* Bring active card above neighbouring cards */
.domain-row:has(.domain-help:hover),
.domain-row:has(.domain-help:focus) {
  z-index:10000 !important;
}

/* Tooltip trigger */
.domain-help {
  position:relative !important;
  z-index:10001 !important;
}

/* Put the balloon BELOW the i icon.
   This prevents it colliding with the top of the card. */
.domain-help::after {
  left:50% !important;
  top:calc(100% + 10px) !important;
  bottom:auto !important;

  transform:translateX(-50%) !important;

  width:270px !important;
  max-width:70vw !important;

  padding:10px 12px !important;

  background:#101828 !important;
  color:#FFFFFF !important;

  border-radius:10px !important;

  box-shadow:
    0 10px 30px rgba(16,24,40,.20) !important;

  z-index:10002 !important;

  white-space:normal !important;
}

/* Arrow now points upward */
.domain-help::before {
  left:50% !important;
  top:calc(100% + 5px) !important;
  bottom:auto !important;

  transform:translateX(-50%) !important;

  border-width:0 5px 5px 5px !important;
  border-style:solid !important;
  border-color:
    transparent
    transparent
    #101828
    transparent !important;

  z-index:10003 !important;
}

/* Ensure result wrappers do not clip the balloon either */
#results-area,
#results-area > div,
#results-area .gradio-html,
#results-area .html-container {
  overflow:visible !important;
}




/* =========================================================
   DOMAIN TOOLTIP — FINAL UNCLIPPED FIX
   ========================================================= */

/* Domain Profile and its Gradio wrappers may not clip tooltips */
#domain-profile,
#domain-profile > div,
#domain-profile .html-container,
#domain-profile .prose,
#domain-profile .domain-grid {
  overflow:visible !important;
}

/* The parent wrappers around Domain Profile must also allow overflow */
#domain-profile,
#domain-profile * {
  contain:none !important;
}

#domain-profile .domain-grid {
  position:relative !important;
  z-index:20 !important;
}

/* Individual domain cards */
#domain-profile .domain-row {
  position:relative !important;
  overflow:visible !important;
  z-index:1 !important;
}

/* Card with active tooltip goes above all neighbouring cards */
#domain-profile .domain-row:has(.domain-help:hover),
#domain-profile .domain-row:has(.domain-help:focus) {
  z-index:9998 !important;
}

/* Info icon */
#domain-profile .domain-help {
  position:relative !important;
  z-index:9999 !important;
}

/* Tooltip balloon */
#domain-profile .domain-help::after {
  z-index:10000 !important;

  top:calc(100% + 10px) !important;
  bottom:auto !important;

  width:300px !important;
  max-width:72vw !important;

  padding:11px 13px !important;

  background:#101828 !important;
  color:#FFFFFF !important;

  border-radius:12px !important;

  white-space:normal !important;
  overflow:visible !important;

  box-shadow:
    0 14px 32px rgba(16,24,40,.22) !important;
}

/* Tooltip arrow */
#domain-profile .domain-help::before {
  z-index:10001 !important;

  top:calc(100% + 5px) !important;
  bottom:auto !important;
}

/* Make the surrounding results hierarchy allow the popup through */
#results-area,
#results-area > div,
#results-area > div > div {
  overflow:visible !important;
}




/* =========================================================
   EAMI HERO — JAZZBERRY / NAVY / TEAL TEXTURE
   No pink
   ========================================================= */

/* Main top EAMI area */
#hero {
  position:relative !important;
  isolation:isolate !important;
  overflow:hidden !important;

  padding:24px 26px !important;
  margin-bottom:18px !important;

  background:
    radial-gradient(
      600px 260px at 3% 0%,
      rgba(165,11,94,.075),
      transparent 48%
    ),
    radial-gradient(
      650px 280px at 52% -10%,
      rgba(41,72,110,.075),
      transparent 48%
    ),
    radial-gradient(
      520px 250px at 100% 10%,
      rgba(28,110,105,.060),
      transparent 46%
    ),
    linear-gradient(
      135deg,
      #FFFFFF 0%,
      #FAFBFC 48%,
      #F6F8FA 100%
    ) !important;

  border-radius:22px !important;
  border:1px solid #E0E5EC !important;

  box-shadow:
    0 10px 30px rgba(15,23,42,.035) !important;
}

/* Fine scientific texture */
#hero::before {
  content:"";
  position:absolute;
  inset:0;
  z-index:-1;
  pointer-events:none;

  background-image:
    radial-gradient(
      circle,
      rgba(31,41,55,.10) .55px,
      transparent .65px
    );

  background-size:18px 18px;

  opacity:.20;

  -webkit-mask-image:
    linear-gradient(
      90deg,
      rgba(0,0,0,.75),
      rgba(0,0,0,.32) 55%,
      rgba(0,0,0,.12)
    );

  mask-image:
    linear-gradient(
      90deg,
      rgba(0,0,0,.75),
      rgba(0,0,0,.32) 55%,
      rgba(0,0,0,.12)
    );
}

/* Very subtle Jazzberry accent line */
#hero::after {
  content:"";
  position:absolute;
  left:26px;
  top:0;

  width:110px;
  height:3px;

  background:linear-gradient(
    90deg,
    #A50B5E,
    #344E6B
  );

  border-radius:0 0 999px 999px;
  opacity:.82;
}

/* Make sure text remains crisp */
#hero h1,
#hero .sub,
#hero .tag,
#hero > div {
  position:relative;
  z-index:2;
}

/* EAMI title */
#hero h1 {
  color:#101828 !important;
}

/* Subtitle */
#hero .sub {
  color:#475467 !important;
}

/* Demo/version badge if this selector is still used */
#hero .tag {
  background:rgba(255,255,255,.80) !important;
  color:#344054 !important;
  border:1px solid rgba(165,11,94,.30) !important;
  backdrop-filter:blur(10px);
  -webkit-backdrop-filter:blur(10px);
}

/* If the current Research Demo badge uses these classes */
#hero .research-badge,
#hero .demo-badge {
  background:rgba(255,255,255,.82) !important;
  color:#344054 !important;
  border:1px solid rgba(165,11,94,.28) !important;

  box-shadow:
    inset 3px 0 0 #A50B5E,
    0 6px 18px rgba(15,23,42,.04) !important;
}

/* Give the workflow strip a slightly cooler surface */
#hero + div,
.workflow-strip,
.hero-flow {
  background:rgba(255,255,255,.78) !important;
  backdrop-filter:blur(8px);
  -webkit-backdrop-filter:blur(8px);
}


"""



def _show_analysis_results():
    return (
        gr.update(visible=False),
        gr.update(visible=True)
    )


def _input_coverage_html(
    sample_type,
    total_reads,
    detected_genes,
    mito,
    line1,
    chimeric,
    ir_index,
    *advanced
):
    """
    Field-population indicator only.
    This is NOT ETIS domain completeness and NOT a quality score.
    Zero is treated as a valid supplied value.
    """

    import math

    n_i = len(IMMUNE)
    n_s = len(SCMC)
    n_b = len(BRCA)
    n_v = len(MT_VARIANTS)
    n_iso = len(ISOFORM_COLS)

    ivals = advanced[:n_i]
    svals = advanced[n_i:n_i+n_s]
    bvals = advanced[n_i+n_s:n_i+n_s+n_b]
    vvals = advanced[n_i+n_s+n_b:n_i+n_s+n_b+n_v]
    iso_vals = advanced[
        n_i+n_s+n_b+n_v:
        n_i+n_s+n_b+n_v+n_iso
    ]

    def available(v):
        if v is None:
            return False
        try:
            return not math.isnan(float(v))
        except (TypeError, ValueError):
            return True

    technical_core = [
        total_reads,
        detected_genes,
        mito,
        line1,
        chimeric,
        ir_index,
    ]

    pathway_values = list(ivals) + list(svals)

    advanced_values = (
        list(bvals)
        + list(vvals)
        + list(iso_vals)
    )

    all_values = (
        technical_core
        + pathway_values
        + advanced_values
    )

    available_total = sum(available(v) for v in all_values)
    total = len(all_values)

    core_available = sum(available(v) for v in technical_core)
    pathway_available = sum(available(v) for v in pathway_values)
    advanced_available = sum(available(v) for v in advanced_values)

    pct = 100.0 * available_total / total if total else 0.0

    if pct >= 90:
        cls = "coverage-high"
    elif pct >= 65:
        cls = "coverage-mid"
    else:
        cls = "coverage-low"

    return f"""
    <div class="input-coverage-card {cls}">
      <div class="coverage-main">
        <div>
          <div class="coverage-label">INPUT DATA COVERAGE</div>
          <div class="coverage-value">
            {available_total}/{total}
            <span>{pct:.0f}%</span>
          </div>
        </div>

        <div class="coverage-track">
          <div class="coverage-fill" style="width:{pct:.1f}%"></div>
        </div>
      </div>

      <div class="coverage-breakdown">
        <span>Core measurements <strong>{core_available}/{len(technical_core)}</strong></span>
        <span>Pathway values <strong>{pathway_available}/{len(pathway_values)}</strong></span>
        <span>Advanced data <strong>{advanced_available}/{len(advanced_values)}</strong></span>
      </div>

      <div class="coverage-note">
        Field-population indicator only · 0 is treated as a supplied value ·
        independent of ETIS Domain completeness
      </div>
    </div>
    """


with gr.Blocks(title="EAMI · Research Demo") as demo:
    gr.HTML("""
    <div id="hero">
      <div>
        <div class="eami-kicker">RESEARCH SOFTWARE PROTOTYPE</div>
        <h1>EAMI</h1>
        <div class="sub">Embryo Arrest Molecular Interpreter</div>
        <div class="hero-description">
          Integrated molecular interpretation of arrested human embryos from long-read transcriptomic data.
        </div>
      </div>

      <div class="demo-badge">
        <strong>RESEARCH DEMO</strong>
        <span>Not for clinical use</span>
      </div>
    </div>

    <div class="eami-workflow">
      <div class="workflow-step">
        <div class="workflow-number">01</div>
        <div>
          <strong>INPUT</strong>
          <span>Long-read transcriptomic features</span>
        </div>
      </div>

      <div class="workflow-arrow">→</div>

      <div class="workflow-step">
        <div class="workflow-number">02</div>
        <div>
          <strong>5 MOLECULAR DOMAINS</strong>
          <span>Mitochondrial · Transcriptomic · RNA processing · Immune/stress · SCMC/DDR</span>
        </div>
      </div>

      <div class="workflow-arrow">→</div>

      <div class="workflow-step">
        <div class="workflow-number">03</div>
        <div>
          <strong>INTERPRETATION</strong>
          <span>Molecular deviation · Profile class · Dominant signature</span>
        </div>
      </div>
    </div>
    """)

    # Compact project/demo controls
    with gr.Row():
        with gr.Column(scale=2):
            sample_type=gr.Radio([("Embryo","embryo"),("Oocyte","oocyte")],value="embryo",label="Material type",elem_id="material-type")
        with gr.Column(scale=3):
            with gr.Accordion("Reference demo loader",open=False,elem_classes=["tight-accordion"],elem_id="reference-loader"):
                with gr.Row():
                    ref_sample=gr.Dropdown(
                        df.loc[df["sample_type"].eq("embryo"), "sample_id"].tolist(),
                        label="Reference sample"
                    )
                    load_btn=gr.Button("Load reference",variant="secondary",elem_id="load-reference")
                load_status=gr.HTML(
            '<div class="sample-status idle">No reference sample loaded</div>',
            elem_id="load-status"
        )

    with gr.Accordion("Report metadata · optional",open=False,elem_id="report-metadata"):
        gr.Markdown("Use pseudonymous labels only.")
        with gr.Row():
            report_sample_label=gr.Textbox(label="Report sample label",placeholder="e.g. EAMI-001")
            report_cycle_id=gr.Textbox(label="Cycle / subject ID",placeholder="optional pseudonymous ID")
        report_notes=gr.Textbox(label="Research notes",lines=2,placeholder="optional")

    gr.HTML('<div class="section-title">Input workspace</div>')

    # Four logical input groups
    with gr.Tabs():
        with gr.Tab("01 · Technical"):
            gr.Markdown("Cohort-position context only — **not** a validated sequencing QC pass/fail rule.")
            with gr.Row():
                total_reads=gr.Number(label="Total reads")
                detected_genes=gr.Number(label="Detected genes")

        with gr.Tab("02 · Molecular core"):
            gr.Markdown("Primary cohort-relative molecular measurements.")
            with gr.Row():
                mito=gr.Number(label="Mitochondrial fraction (% chrM)")
                line1=gr.Number(label="LINE-1 activity (RPM)")
            with gr.Row():
                chimeric=gr.Number(label="Chimeric reads (RPM)")
                ir_index=gr.Number(label="Intron Retention Index")

        with gr.Tab("03 · Pathways"):
            immune_components=[]
            scmc_components=[]
            with gr.Accordion("Immune / stress panel",open=True,elem_id="immune-panel"):
                gr.Markdown("0–1 cohort-scaled values.")
                for i in range(0,len(IMMUNE),3):
                    with gr.Row():
                        for g in IMMUNE[i:i+3]:
                            immune_components.append(gr.Slider(0,1,value=0,step=.001,label=g))
            with gr.Accordion("SCMC / DDR panel",open=False,elem_id="scmc-panel"):
                gr.Markdown("0–1 cohort-scaled values.")
                for i in range(0,len(SCMC),3):
                    with gr.Row():
                        for g in SCMC[i:i+3]:
                            scmc_components.append(gr.Slider(0,1,value=0,step=.001,label=g))

        with gr.Tab("04 · Advanced"):
            brca_components=[]
            isoform_components=[]
            variant_components=[]

            with gr.Accordion("BRCA1 isoforms",open=True,elem_id="brca-accordion"):
                with gr.Row():
                    for g,label in zip(BRCA,["ENST00000354071.8","ENST00000618469.2","ENST00000634433.2"]):
                        brca_components.append(gr.Number(value=0,label=f"{label} (FPKM)"))

            with gr.Accordion("Mitochondrial variants",open=False,elem_id="mtvar-accordion"):
                gr.Markdown(
                    "**Transcript-derived variant fractions (0–1).** "
                    "The current quantified panel contains seven exact Patient-8 positions."
                )
                for i in range(0,len(MT_VARIANTS),3):
                    with gr.Row():
                        for g,label in zip(MT_VARIANTS[i:i+3],MT_VARIANT_LABELS[i:i+3]):
                            variant_components.append(
                                gr.Number(
                                    value=None,
                                    minimum=0,
                                    maximum=1,
                                    step=0.01,
                                    precision=3,
                                    label=f"{label} · variant fraction"
                                )
                            )

            with gr.Accordion("Isoform switching · 25-transcript panel",open=False,elem_id="isoform-accordion"):
                gr.Markdown(
                    "Approximate 0–1 reference reconstruction from the supplied heatmap. "
                    "Leave blank for a new sample when not measured."
                )
                for i in range(0,len(ISOFORM_COLS),3):
                    with gr.Row():
                        for g,label in zip(ISOFORM_COLS[i:i+3],ISOFORM_LABELS[i:i+3]):
                            isoform_components.append(gr.Number(value=None,minimum=0,maximum=1,label=label))

            with gr.Accordion("Immune pathway reference context",open=False,elem_id="immune-ref-accordion"):
                gr.Markdown(
                    "Cohort-wide detected-gene counts only; **not used in classification**.\n\n"
                    "| Group | Genes |\n|---|---:|\n"
                    "| Chemokines | 63 |\n| TNF Superfamily | 49 |\n| Interleukins | 46 |\n"
                    "| Interferons | 43 |\n| NLR (Inflammasome/Sensor) | 24 |\n| Caspases | 12 |\n"
                    "| Signaling Adapters | 11 |\n| TLR | 10 |\n| NF-κB | 5 |\n"
                    "| Inflammasome Core | 4 |\n| RLR | 2 |\n| cGAS-STING | 2 |\n| NOD-like NLR | 2 |"
                )

    input_coverage = gr.HTML(
        """
        <div class="input-coverage-card coverage-idle">
          <div class="coverage-label">INPUT DATA COVERAGE</div>
          <div class="coverage-idle-text">
            Load or enter a sample to assess available input fields.
          </div>
        </div>
        """,
        elem_id="input-coverage"
    )

    # Sticky workflow actions
    with gr.Row(elem_id="action-bar"):
        analyze_btn=gr.Button("ANALYZE SAMPLE",variant="primary",elem_id="analyze-btn",scale=3)
        clear_btn=gr.Button(
            "NEW SAMPLE",
            variant="secondary",
            elem_id="clear-btn",
            scale=1
        )
        report_btn=gr.Button(
            "GENERATE REPORT",
            variant="secondary",
            elem_id="report-btn",
            scale=2
        )

    analysis_placeholder = gr.HTML(
        """
        <div class="analysis-empty-state">
          <div class="empty-state-mark">EAMI</div>
          <div class="empty-state-copy">
            <div class="empty-state-title">
              No molecular interpretation generated yet
            </div>
            <div class="empty-state-text">
              Load a reference sample or enter molecular data, then select
              <strong>Analyze Sample</strong>.
            </div>
          </div>
        </div>
        """,
        visible=True,
        elem_id="analysis-placeholder"
    )

    with gr.Group(
        visible=False,
        elem_id="results-area"
    ) as results_group:
        gr.HTML('<div class="section-title">Executive molecular summary</div>')

        # Executive hierarchy: four most important outputs
        with gr.Row(elem_id="exec-summary-row"):
            etis_box=gr.HTML()
            state_box=gr.HTML()
            phenotype_box=gr.HTML()
            completeness_box=gr.HTML()

        gr.HTML('<div class="section-title">Domain profile</div>')
        domain_profile=gr.HTML(elem_id="domain-profile")

        gr.HTML('<div class="section-title">Molecular interpretation</div>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                plot=gr.Plot(label="Molecular Signal Map",show_label=False)
            with gr.Column(scale=5):
                key_findings=gr.Markdown(elem_id="key-findings")

        gr.HTML('<div class="section-title">Supporting evidence</div>')
        with gr.Row():
            reference_box=gr.HTML()
            technical_box=gr.HTML()
            isoform_box=gr.HTML()

        with gr.Accordion("Detailed explainability & audit trail",open=False,elem_id="audit-trail"):
            detailed_explanation=gr.Markdown()
            with gr.Accordion("ETIS domain scoring table",open=False):
                domain_table=gr.Dataframe(interactive=False,elem_id="domain-table")
            with gr.Accordion("Closest cohort profiles",open=False):
                table=gr.Dataframe(interactive=False,elem_id="cohort-table")

    # Report downloads intentionally placed at the end and kept compact
    gr.HTML('<div class="section-title">Molecular Interpretation Report</div>')
    with gr.Group(elem_id="report-downloads"):
        report_summary=gr.Markdown("**Report export** · Generate the molecular interpretation report to enable downloads.",elem_id="report-summary")
        with gr.Row():
            html_download=gr.DownloadButton(
                "HTML REPORT",
                value=None,
                variant="secondary",
                interactive=False,
                elem_id="html-download",
                scale=1,
            )
            pdf_download=gr.DownloadButton(
                "PDF REPORT",
                value=None,
                variant="primary",
                interactive=False,
                elem_id="pdf-download",
                scale=1,
            )
            gr.Markdown("Generated reports are saved locally in the EAMI `reports/` directory.",container=False)

    gr.Markdown(
        "**Research Use Only.** Not for embryo selection, transfer decisions, diagnosis, "
        "viability assessment or causal attribution.",
        elem_id="ruo-footer"
    )

    gr.HTML(
        """
        <div class="eami-footer">
          <div class="eami-footer-brand">
            <strong>EAMI</strong>
            <span>Embryo Arrest Molecular Interpreter</span>
          </div>

          <div class="eami-footer-meta">
            <span>Research Demo · v0.8.2</span>
            <span class="footer-dot">•</span>
            <span>PhD-derived reference cohort · 16 samples from 7 patients</span>
            <span class="footer-dot">•</span>
            <span>Long-read transcriptomic interpretation</span>
          </div>

          <div class="eami-footer-ruo">
            Research Use Only · Not for clinical decision-making
          </div>
        </div>
        """,
        elem_id="eami-footer"
    )

    # Preserve original parser order
    all_inputs=[
        sample_type,total_reads,detected_genes,mito,line1,chimeric,ir_index
    ]+immune_components+scmc_components+brca_components+variant_components+isoform_components

    sample_type.change(
        fn=filter_reference_samples,
        inputs=[sample_type],
        outputs=[ref_sample]
    )

    load_event = load_btn.click(
        load_reference,
        [ref_sample],
        all_inputs
    )

    load_event.then(
        fn=lambda: '<div class="sample-status ready">✓ Sample loaded · Ready for analysis</div>',
        inputs=None,
        outputs=load_status
    )

    load_event.then(
        fn=_input_coverage_html,
        inputs=all_inputs,
        outputs=input_coverage
    )

    ref_sample.change(
        fn=lambda: '<div class="sample-status idle">Reference selected · Press Load reference</div>',
        inputs=None,
        outputs=load_status
    )

    clear_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { window.location.reload(); }"
    )

    analyze_btn.click(
        fn=_input_coverage_html,
        inputs=all_inputs,
        outputs=input_coverage
    )

    analyze_btn.click(
        fn=_show_analysis_results,
        inputs=None,
        outputs=[analysis_placeholder, results_group]
    )

    analyze_btn.click(
        analyze_ui,
        all_inputs,
        [
            etis_box,state_box,phenotype_box,completeness_box,
            reference_box,technical_box,isoform_box,domain_profile,
            domain_table,table,plot,detailed_explanation
        ]
    ).then(
        lambda *x: _key_findings_md(_analysis_bundle(*x)) if _analysis_bundle(*x) else "Insufficient values.",
        all_inputs,
        key_findings
    )

    report_btn.click(
        generate_report_ui,
        [report_sample_label,report_cycle_id,report_notes]+all_inputs,
        [report_summary,html_download,pdf_download]
    )

if __name__=="__main__":
    demo.launch(css=CSS, theme=gr.themes.Base(), server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
