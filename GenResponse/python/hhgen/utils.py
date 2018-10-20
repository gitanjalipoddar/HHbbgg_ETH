import numpy as np
import skhep.math as skp

## -----------------------------------------------------------------------------------------
def calc_p4extra(df,prefix):
    px = df[prefix+"_px"]
    py = df[prefix+"_py"]
    pz = df[prefix+"_pz"]
    en = df[prefix+"_e"]

    df[prefix+"_pt"] = np.sqrt( px**2 + py**2 )
    df[prefix+"_eta"] = np.arcsinh( pz / df[prefix+"_pt"] )
    df[prefix+"_phi"] = np.arcsin( py / df[prefix+"_pt"] )
    df[prefix+"_m"] = np.sqrt( en**2 - px**2 -py**2 -pz**2 )

## -----------------------------------------------------------------------------------------
def calc_sump4(df,dest,part1,part2): 
    for comp in "_px","_py","_pz","_e":
        df[dest+comp] = df[part1+comp] + df[part2+comp]
    calc_p4extra(df,dest)

## -----------------------------------------------------------------------------------------
def calc_cos_theta_cs(df):
    ebeam = 6.5 #units TeV
    
    def cos_theta_cs(X):
        hh=skp.LorentzVector(X["hh_px"],X["hh_py"],X["hh_pz"],X["hh_e"])
        booster= hh.boostvector
        #boosting p1, p2 and hgg according to boost of hh, and converting them to unit vectors
        p1=skp.LorentzVector(0,0,ebeam,ebeam)
        p1_boost=p1.boost(booster).vector.unit()
        p2=skp.LorentzVector(0,0,-ebeam,ebeam)
        p2_boost= p2.boost(booster).vector.unit()
        hgg=skp.LorentzVector(X["hgg_px"],X["hgg_py"],X["hgg_pz"],X["hgg_e"])
        hgg_boost=hgg.boost(booster).vector.unit()
        CSaxis=(p1_boost-p2_boost).unit() #bisector
        return np.cos(CSaxis.angle(hgg_boost))

    df["cos_theta_cs"]= df.apply(cos_theta_cs,axis=1)   
    
## -----------------------------------------------------------------------------------------
def calc_cos_theta(df,part1,part2):
    
    def cos_theta(X): 
        booster=skp.LorentzVector(X[part1+"_px"],X[part1+"_py"],X[part1+"_pz"],X[part1+"_e"])
        boosted=skp.LorentzVector(X[part2+"_px"],X[part2+"_py"],X[part2+"_pz"],X[part2+"_e"])
        #boosting the leadJet/leadPhoton according to boost of hbb/hgg 
        boosted_boost=boosted.boost(booster.boostvector).vector.unit()
        booster=booster.vector.unit() 
        return np.cos(boosted_boost.angle(booster)) #angle between boost and boosted leadJet/leadPho
       
    df["cos_theta_"+part1]= df.apply(cos_theta,axis=1)
