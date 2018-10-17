import numpy as np

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
def calc_cos_theta(df, part):
    df["cos_theta_"+part] = np.cos(2*np.arctan(np.exp(-df["h"+part+"_eta"])))

## -----------------------------------------------------------------------------------------
def calc_cos_theta_cs(df):
    num=np.abs(np.sinh(df["hh_delta_eta"]))*2.*df["hgg_pt"]*df["hbb_pt"]
    den=np.sqrt(1+(df["hh_pt"]/df["hh_m"])**2.)*(df["hh_m"]**2)
    df["cos_theta_cs"]=num/den
    

#def cos_theta_hlx(leadPho,subleadPho,leadJ,subleadJ):
#def cos_theta_cs(diPho,diJet):