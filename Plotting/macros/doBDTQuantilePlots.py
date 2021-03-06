#!/usr/bin/env python

# Standard python import
from optparse import OptionParser, make_option
import fnmatch, glob, os, sys, json, itertools, array
from pprint import pprint
#sys.argv.append( '-b' )
from ROOT import *
gROOT.SetBatch(True)
gROOT.ProcessLine(".L ~/setTDRStyle.C");
setTDRStyle()
gStyle.SetOptStat(0)

BDTbins=[]
weight = 'weight'
variables={}
savefmts=['.png','.root','.pdf','.jpg']
colors=[kRed, kBlue, kGreen, kOrange, kCyan, kMagenta, kYellow, kGray, kPink-7, kOrange+1]

#quant_x = [0., 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.]
quant_x = [0., 0.20, 0.40, 0.60,  0.80, 1.]

categories={}
categories['sig']=str('reducedTree_sig')
categories['dipho']=str('reducedTree_bkg_0')
categories['ggH']=str('reducedTree_bkg_3')
categories['ttH']=str('reducedTree_bkg_7')


#mva input variables
variables['customLeadingPhotonIDMVA']     = dict(name='leadPhoMva',nbins=120,xmin=-1.,xmax=1., xaxis=" photon ID MVA, leading #gamma", yaxis='a.u.', line='', logy=0, logx=0, leg="m")
variables['customSubLeadingPhotonIDMVA']  = dict(name='subleadPhoMva',nbins=120,xmin=-1.,xmax=1., xaxis=" photon ID MVA, subleading #gamma", yaxis='a.u.', line='', logy=0, logx=0, leg="m")
variables['fabsCosThetaStarCS']    = dict(name='cosThetaStar',nbins=100,xmin=0.,xmax=1, xaxis="cos(#Theta_{*})", yaxis='a.u.', line='', logy=0, logx=0, leg="m")
variables['fabsCosThetabb']    = dict(name='cosThetabb',nbins=100,xmin=0.,xmax=1, xaxis="cos(#Theta_{bb})", yaxis='a.u.', line='', logy=0, logx=0, leg="m")
variables['fabsCosThetagg']    = dict(name='cosThetagg',nbins=100,xmin=0.,xmax=1, xaxis="cos(#Theta_{gg})", yaxis='a.u.', line='', logy=0, logx=0, leg="m")
variables['leadingJetbDis']    = dict(name='leadJetbTag',nbins=100,xmin=0.,xmax=1, xaxis="btag , leading Jet", yaxis='a.u.', line='', logy=1, logx=0, leg="r")
variables['subleadingJetbDis']    = dict(name='subLeadJetbTag',nbins=100,xmin=0.,xmax=1, xaxis="btag , subleading Jet", yaxis='a.u.', line='', logy=1, logx=0, leg="r")
variables['diphotonPtOverdiHiggsM']    = dict(name='diphoPtOverM',nbins=50,xmin=0.,xmax=2.5, xaxis="p_{T}^{#gamma#gamma}/m_{HH}", yaxis='a.u.', line='', logy=0, logx=0, leg="r")
variables['dijetPtOverdiHiggsM']    = dict(name='dijetPtOverM',nbins=50,xmin=0.,xmax=2.5, xaxis="p_{T}^{jj}/m_{HH}", yaxis='a.u.', line='', logy=0, logx=0, leg="r")
variables['subleadingPhotonSigOverE']    = dict(name='subleadSigOverE',nbins=50,xmin=0.,xmax=0.1, xaxis="#sigma_{E}/E, subleading Photon", yaxis='a.u.', line='', logy=1, logx=0, leg="r")
variables['leadingPhotonSigOverE']    = dict(name='leadSigOverE',nbins=50,xmin=0.,xmax=0.1, xaxis="#sigma_{E}/E, leading Photon", yaxis='a.u.', line='', logy=1, logx=0, leg="r")



variables['sigmaMOverMDecorr'] = dict(name='sigmaMOverM',nbins=50,xmin=0.,xmax=0.1, xaxis="#sigma_{M}/M", yaxis='a.u.', line='', logy=1, logx=0, leg="r")
variables['DiJetDiPhoDR'] = dict(name='diJetDiPhoDR',nbins=50,xmin=0.,xmax=6, xaxis="#DeltaR(#gamma#gamma,bb)", yaxis='a.u.', line='', logy=0, logx=0, leg="r")

variables['MVAOutput']         = dict(name='MVA',nbins=600,xmin=0.,xmax=1., xaxis="BDT output", yaxis='a.u.', line='', logy=1, logx=0, leg="m")
#mass variables

variables['Mgg']        = dict(name='mgg',nbins=80,xmin=100.,xmax=180., xaxis="m_{#gamma#gamma}", yaxis='a.u.', line='', logy=0, logx=0, leg="r")
variables['Mjj']        = dict(name='mjj',nbins=80,xmin=70.,xmax=180., xaxis="m_{bb}", yaxis='a.u.', line='', logy=0, logx=0, leg="r")
variables['MX']        = dict(name='mx',nbins=80,xmin=200.,xmax=900., xaxis="M_{X}", yaxis='a.u.', line='', logy=0, logx=0, leg="r")

# Main routine
def main(o,args):
    if 0 and options.sigFile.startswith("/store"):
        filepath = "root://eoscms/"+str(options.sigFile)
    else:
        filepath = options.sigFile
    print 'filepath is '+str(filepath)
    infile = TFile.Open(filepath)
    defineBDTBins(infile)
    print BDTbins
    outFullPath=options.outdir+'/'+options.outfile.replace(".root","")

    if not os.path.exists(outFullPath):
        os.makedirs(outFullPath)




    histos={}
    categs=categories.keys()

    
    if options.doQuantilesPlots:
        outfile = TFile(outFullPath+'/'+options.outfile, 'RECREATE')
    #draw and save plots
        histos= getPlots(infile)        
        outfile.cd()
        for histo in histos:
            histo.Write()
            c = TCanvas('c','',1)
            histo.Draw('ehist')
            print "dddddddddddddddddddddddddddddddddddddddddddd"
            c.SetLogy(variables[str(histo.GetName()).split('_')[1]]['logy'])
            c.SetLogx(variables[str(histo.GetName()).split('_')[1]]['logx'])
            for fmt in ['.png','.root']:
                c.SaveAs(str(options.outdir)+'/'+options.outfile.replace(".root","")+"/"+str(histo.GetName())+str(fmt))

        outfile.Write()
        outfile.Close()

    plotFile = TFile(outFullPath+'/'+options.outfile, 'READ')
    stacks={}
    outStackFile = TFile(outFullPath+'/'+options.outfile.replace(".root","")+"_stacks.root", 'RECREATE')

    ll = TLegend(0.1,0.65,0.35,0.9)   
    lm = TLegend(0.325,0.65,0.575,0.9)   
    lr = TLegend(0.65,0.65,0.9,0.9)   
    ll.SetBorderSize(0)
    lm.SetBorderSize(0)
    lr.SetBorderSize(0)

    for variable in variables.keys():
        for cat in categs:
            stacks[str(variable)+'_'+str(cat)]= THStack(str(variable)+'_'+str(cat), '')
            for i in reversed(range(len(BDTbins))):
                bin = BDTbins[i]
                name = 'hist_'+str(variable)+'_'+str(cat)+'_BDT'+str(bin[0]).replace('.','').replace('-','m')+'to'+str(bin[1]).replace('.','').replace('-','m')
                print name
                histo = plotFile.Get(str(name))
                if histo.Integral() != 0:
                    histo.Print()
                    if cat == 'ggH': 
                        histo.Rebin(4)
                    if not variable == options.quantileVar:
                        histo.Scale(1./histo.Integral())
                    print ''
                    histo.SetLineColor(colors[i])
                    histo.SetFillColor(colors[i])
#                    if variable == 'Mgg' or variable == 'Mjj':
#                        histo.SetFillStyle(3004)
                    histo.SetFillStyle(3004)
                    stacks[str(variable)+'_'+str(cat)].Add(histo)
                    if cat == categs[0] and variable == variables.keys()[0]:
                        ll.AddEntry(histo, str(quant_x[i+1])+'< #epsilon_{sig} <'+str(quant_x[i]),'f')
                        lm.AddEntry(histo, str(quant_x[i+1])+'< #epsilon_{sig} <'+str(quant_x[i]),'f')
                        lr.AddEntry(histo, str(quant_x[i+1])+'< #epsilon_{sig} <'+str(quant_x[i]),'f')



    outStackFile.cd()

    for stack in stacks.values():
        stack.Write()
        c = TCanvas('c','',1)
        
#        stack.Draw('ehist nostackb')
        print str(stack.GetName()).split('_')[0]
#        if str(stack.GetName()).split('_')[0] != 'Mgg' and str(stack.GetName()).split('_')[0] != 'Mjj':
#            stack.Draw('ehist')
#        else:
#            stack.Draw('hist nostack')
        stack.Draw('hist nostack')

        stack.GetXaxis().SetTitle(str(variables[str(stack.GetName()).split('_')[0]]['xaxis']))
        stack.GetYaxis().SetTitle(str(variables[str(stack.GetName()).split('_')[0]]['yaxis']))
#        stack.Draw('ehist nostackb')
#        stack.Draw('ehist')
        #l.DrawBox( variables[str(stack.GetName()).split('_')[0]]['leg'][0], variables[str(stack.GetName()).split('_')[0]]['leg'][1], variables[str(stack.GetName()).split('_')[0]]['leg'][2], variables[str(stack.GetName()).split('_')[0]]['leg'][3])
        if str(variables[str(stack.GetName()).split('_')[0]]['leg']) == 'l':
            ll.Draw()
        if str(variables[str(stack.GetName()).split('_')[0]]['leg']) == 'm':
            lm.Draw()
        if str(variables[str(stack.GetName()).split('_')[0]]['leg']) == 'r':
            lr.Draw()
        c.SetLogy( variables[str(stack.GetName()).split('_')[0]]['logy'] )
        c.SetLogx( variables[str(stack.GetName()).split('_')[0]]['logx'] )
        tex_m=TLatex()
        tex_m.SetNDC()
        tex_m.SetTextAlign(12)
        tex_m.SetTextFont(42)
        tex_m.SetTextSize(0.035)
        tex_m.SetLineWidth(2)
        tex_m.DrawLatex(0.125,0.969,"#scale[1.1]{CMS} Preliminary")
        tex_m.DrawLatex(0.400,0.969,"#font[12]{Simulation}")
        
        tex_m=TLatex()
        tex_m.SetNDC()
        tex_m.SetTextAlign(12)
        tex_m.SetTextFont(42)
        tex_m.SetTextSize(0.04)
        #        tex_m.SetLineWidth(2)
        tex_m.DrawLatex(0.83,0.969,"13TeV")
        for fmt in savefmts:
#            c.SaveAs('stack_'+str(stack.GetName())+str(fmt))
#            c.SaveAs('stack_'+str(stack.GetName())+'_rm1stWorstQuant'+str(fmt))
#            c.SaveAs('stack_'+str(stack.GetName())+'_nostack'+str(fmt))
            c.SaveAs(str(options.outdir)+'/'+options.outfile.replace(".root","")+"/"'stack_'+str(stack.GetName())+str(fmt))
###            c.SaveAs('stack_'+str(stack.GetName())+'_nostackzoom'+str(fmt))

    outStackFile.Write()
    outStackFile.Close()


def defineBDTBins(infile):
    t=infile.Get(str(options.treename))
    hist = TH1D('h_bdt','h_bdt', 600, options.min_quantileVar, options.max_quantileVar)
    hist.Sumw2()
    t.Project(hist.GetName(), options.quantileVar,str(weight))
    quant_x_array = array.array('d', quant_x)
    quantiles_array = array.array('d',[0.0]*len(quant_x_array))
    nn = hist.GetQuantiles(len(quant_x_array), quantiles_array, quant_x_array)
    for i in range(len(quantiles_array)-1):
        couple = [quantiles_array[i],quantiles_array[i+1]]
        BDTbins.append(couple)


def getPlots(infile):
    plots=[]
    #        t.Print()
    for variable in variables.keys():
        for bin in BDTbins:
            for cat in categories.keys():
                t=infile.Get(categories[cat])
                hist = TH1D('hist_'+str(variable)+'_'+str(cat)+'_BDT'+str(bin[0]).replace('.','').replace('-','m')+'to'+str(bin[1]).replace('.','').replace('-','m'),'hist_'+str(variable)+'_'+str(cat)+'_BDT'+str(bin[0]).replace('.','').replace('-','m')+'to'+str(bin[1]).replace('.','').replace('-','m'), variables[str(variable)]['nbins'], variables[str(variable)]['xmin'],variables[str(variable)]['xmax'])
                hist.Sumw2()
#            htemp = TH1D('htemp','htemp', variables[str(variable)]['nbins'], variables[str(variable)]['xmin'],variables[str(variable)]['xmax'])
                cut = str(weight)+'*('+options.quantileVar+' >'+str(bin[0])+' && '+options.quantileVar+'<= '+str(bin[1])+')'
                print cut
                t.Project(hist.GetName(), str(variable), str(cut))
                hist.GetXaxis().SetTitle(str(variables[str(variable)]['xaxis']))
                print hist.GetEntries()
            #TLatex
                plots.append(hist)
    return plots



## ------------------------------------------------------------------------------------------------------------------------------------------------------    
if __name__ == "__main__":
    parser = OptionParser(option_list=[
            make_option("-s", "--sigFile",
                        action="store", type="string", dest="sigFile",
                        default="tmvadiPho_Moriond17.root",
                        help="pattern of sigFile to be read", metavar="PATTERN"
                        ), 
            make_option("-t", "--treeName",
                        action="store", type="string", dest="treename",
                        default="reducedTree_sig",
                        help="TTree name", metavar="TREENAME"
                        ),
            make_option("-o", "--outfile",
                        action="store", type="string", dest="outfile",
                        default="tmvaPlots_Moriond17_noQCD.root",
                        help="outputfile", metavar="FILE"
                        ),
            make_option("-D", "--outputdir",
                        action="store_true", dest="outdir",
                        default="plots",
                        ),
            make_option("-q", "--quantileVar",
                        action="store", type="string", dest="quantileVar",
                        default="MVAOutput",
                        help="Variable to define quantile",
                        ),
            make_option("-m", "--MinQuantileVar",
                        action="store", type="float", dest="min_quantileVar",
                        default=0,
                        help="Min value of quantile variable",
                        ),
            make_option("-M", "--MaxQuantileVar",
                        action="store", type="float", dest="max_quantileVar",
                        default=1,
                        help="Max value of quantile variable",
                        ),

            make_option("-k", "--skipQuantilePlots",
                        action="store_false", dest="doQuantilesPlots",
                        default=True,
                        help="do quantile plots"
                        ),
            ]
                          )

    (options, args) = parser.parse_args()

#    sys.argv.append("-b")
    pprint(options.__dict__)

#    import ROOT
    sys.argv.append("-b")
    main(options, args)



