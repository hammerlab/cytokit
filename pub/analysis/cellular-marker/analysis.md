Cytokit Cellular Marker Profiling
================

This analysis contains the gating workflow used to process 8 pan-T Cell imaging experiments as well as a comparison of the size of the inferred CD4+ and CD8+ populations to results from a flow cytometer.

``` r
library(tidyverse)
library(flowCore)
library(openCyto)
library(ggcyto)

data_dir <- '/Volumes/disk1/cytokit/cellular-marker'

# Define experiment names and variants to run analysis for
experiments <- c(
  '20180614_D22_RepA_Tcell_CD4-CD8-DAPI_5by5',
  '20180614_D22_RepB_Tcell_CD4-CD8-DAPI_5by5',
  '20180614_D23_RepA_Tcell_CD4-CD8-DAPI_5by5',
  '20180614_D23_RepB_Tcell_CD4-CD8-DAPI_5by5',
  '20181116-d40-r1-20x-5by5',
  '20181116-d40-r2-20x-5by5',
  '20181116-d41-r1-20x-5by5',
  '20181116-d41-r2-20x-5by5'
)
variants <- c('v00')
```

Load Imaging Data
-----------------

First, generate metadata (`gsm` = "Gating Set Metadata") containing the sample (aka experiment) names, replicate identifiers, and "variants", where a variant corresponds to a Cytokit configuration used in processing.

``` r
gsm <- expand.grid(experiments, variants) %>% set_names(c('experiment', 'variant')) %>%
  mutate(path=str_glue('{data_dir}/{experiment}/output/{variant}/cytometry/data.fcs')) %>%
  mutate(donor=str_to_upper(str_extract(experiment, '[dD]\\d{2}'))) %>%
  mutate(replicate=str_extract(experiment, 'Rep[AB]|r[12]')) %>%
  mutate(replicate=case_when(
    replicate == 'r1' ~ 'RepA',
    replicate == 'r2' ~ 'RepB',
    TRUE ~ replicate
  )) %>% 
  mutate(sample=str_glue('{donor}_{replicate}_{variant}')) %>%
  as('AnnotatedDataFrame')
sampleNames(gsm) <- gsm@data$sample
gsm@data %>% knitr::kable()
```

|                | experiment                                     | variant | path                                                                                                                | donor | replicate | sample         |
|----------------|:-----------------------------------------------|:--------|:--------------------------------------------------------------------------------------------------------------------|:------|:----------|:---------------|
| D22\_RepA\_v00 | 20180614\_D22\_RepA\_Tcell\_CD4-CD8-DAPI\_5by5 | v00     | /Volumes/disk1/cytokit/cellular-marker/20180614\_D22\_RepA\_Tcell\_CD4-CD8-DAPI\_5by5/output/v00/cytometry/data.fcs | D22   | RepA      | D22\_RepA\_v00 |
| D22\_RepB\_v00 | 20180614\_D22\_RepB\_Tcell\_CD4-CD8-DAPI\_5by5 | v00     | /Volumes/disk1/cytokit/cellular-marker/20180614\_D22\_RepB\_Tcell\_CD4-CD8-DAPI\_5by5/output/v00/cytometry/data.fcs | D22   | RepB      | D22\_RepB\_v00 |
| D23\_RepA\_v00 | 20180614\_D23\_RepA\_Tcell\_CD4-CD8-DAPI\_5by5 | v00     | /Volumes/disk1/cytokit/cellular-marker/20180614\_D23\_RepA\_Tcell\_CD4-CD8-DAPI\_5by5/output/v00/cytometry/data.fcs | D23   | RepA      | D23\_RepA\_v00 |
| D23\_RepB\_v00 | 20180614\_D23\_RepB\_Tcell\_CD4-CD8-DAPI\_5by5 | v00     | /Volumes/disk1/cytokit/cellular-marker/20180614\_D23\_RepB\_Tcell\_CD4-CD8-DAPI\_5by5/output/v00/cytometry/data.fcs | D23   | RepB      | D23\_RepB\_v00 |
| D40\_RepA\_v00 | 20181116-d40-r1-20x-5by5                       | v00     | /Volumes/disk1/cytokit/cellular-marker/20181116-d40-r1-20x-5by5/output/v00/cytometry/data.fcs                       | D40   | RepA      | D40\_RepA\_v00 |
| D40\_RepB\_v00 | 20181116-d40-r2-20x-5by5                       | v00     | /Volumes/disk1/cytokit/cellular-marker/20181116-d40-r2-20x-5by5/output/v00/cytometry/data.fcs                       | D40   | RepB      | D40\_RepB\_v00 |
| D41\_RepA\_v00 | 20181116-d41-r1-20x-5by5                       | v00     | /Volumes/disk1/cytokit/cellular-marker/20181116-d41-r1-20x-5by5/output/v00/cytometry/data.fcs                       | D41   | RepA      | D41\_RepA\_v00 |
| D41\_RepB\_v00 | 20181116-d41-r2-20x-5by5                       | v00     | /Volumes/disk1/cytokit/cellular-marker/20181116-d41-r2-20x-5by5/output/v00/cytometry/data.fcs                       | D41   | RepB      | D41\_RepB\_v00 |

Next, load the FCS file for each experiment with special considerations for experiments where particular tiles were found to have large amounts of oversaturation:

``` r
load_fcs <- function(path, donor, replicate) {
  fr <- read.FCS(path, column.pattern='PHA', invert.pattern = TRUE)
  if (donor == 'D22' && replicate == 'RepB'){
    d <- exprs(fr)
    mask <- d[,'tilex'] < 4 | d[,'tiley'] > 2
    cat(sprintf('Removing %s rows of %s for file %s\n', sum(!mask), nrow(fr), path))
    fr <- Subset(fr, mask)
  }
  if (donor == 'D23' && replicate == 'RepB'){
    d <- exprs(fr)
    mask <- (d[,'tilex'] != 1 | d[,'tiley'] != 1) & (d[,'tilex'] != 1 | d[,'tiley'] != 2)
    cat(sprintf('Removing %s rows of %s for file %s\n', sum(!mask), nrow(fr), path))
    fr <- Subset(fr, mask)
  }
  fr
}

# Create a named list of flowFrames
fsr <- gsm@data %>% select(path, donor, replicate) %>% 
  pmap(load_fcs) %>% set_names(gsm@data$sample)
```

    ## Removing 949 rows of 6631 for file /Volumes/disk1/cytokit/cellular-marker/20180614_D22_RepB_Tcell_CD4-CD8-DAPI_5by5/output/v00/cytometry/data.fcs
    ## Removing 939 rows of 7794 for file /Volumes/disk1/cytokit/cellular-marker/20180614_D23_RepB_Tcell_CD4-CD8-DAPI_5by5/output/v00/cytometry/data.fcs

``` r
# Compile frames into single flowSet with appropriate metadata
fsr <- flowSet(fsr)
sampleNames(fsr) <- gsm@data$sample
phenoData(fsr) <- gsm

# Specify biexp transforms for CD4/CD8 only
chnl <- c("ciCD4", "ciCD8")
trans <- transformList(chnl, biexponentialTransform())
fst <- transform(fsr, trans)

# Initialize a gating set with the appropriate marker names
gs <- GatingSet(fst)
markernames(gs) <- fsr@colnames %>% set_names(fsr@colnames)
gs
```

    ## A GatingSet with 8 samples

Apply Gating
------------

Before trying to isolate CD4/CD8 cells, build a workflow that also uses the DAPI intensity and some cell morphology properties to remove cells that are likely the result of illumination artifacts or poor nuclei segmentation:

``` r
add_pop(
  gs, alias="dapi", pop="+", parent='root', dims='niDAPI,ciDAPI', 
  gating_method='flowClust.2d', gating_args="K=1"
)
add_pop(
  gs, alias="circularity", pop="+", parent='dapi', dims='cellcircularity,nucleuscircularity', 
  gating_method='flowClust.2d', gating_args='K=1'
)
add_pop(
  gs, alias="diameter", pop="+", parent='circularity', dims='celldiameter,nucleusdiameter', 
  gating_method='flowClust.2d', gating_args='K=1'
)
add_pop(
  gs, alias="*", pop="*", parent='diameter', dims='ciCD4,ciCD8', 
  gating_method='quadGate.tmix', gating_args="K=3"
)
recompute(gs)
plot(gs)
```

![](analysis_files/figure-markdown_github/unnamed-chunk-6-1.png)

Resulting Populations
---------------------

Show all of the cell filtering populations leading up to the final CD4/CD8 isolation step:

``` r
flowWorkspace.par.set("plotGate", list(xlim = "data", ylim = "data"))
plot_gate <- function(node){
  plotGate(
    gs, node, layout=c(4,2), 
    par.settings=list(
      gate=list(col = "black", lwd=1),
      panel.background = list(col = "white"),
      strip.background = list(col = "white"),
      axis.text = list(cex = .5)
    ),
    xbin = 0,
    marker.only = TRUE,
    scales=list(alternating=TRUE)
  )
}
plot_gate('dapi')
```

![](analysis_files/figure-markdown_github/unnamed-chunk-7-1.png)

``` r
plot_gate('circularity')
```

![](analysis_files/figure-markdown_github/unnamed-chunk-7-2.png)

``` r
plot_gate('diameter')
```

![](analysis_files/figure-markdown_github/unnamed-chunk-7-3.png)

Show quadrant gating on CD4/CD8 inferred via t mixture model:

``` r
nodes <- getNodes(gs, path=1)[5:8]
p_xy <- plotGate(
  gs, nodes, 
  stats=FALSE, 
  layout=c(4,2), 
  par.settings=list(
    gate=list(col = "black", lwd=1),
    panel.background = list(col = "white"),
    strip.background = list(col = "white"),
    axis.text = list(cex = .5),
    axis.line = list(col = 'gray')
  ),
  xlim = c(3.25, 6), ylim = c(3.5, 6.5), 
  xbin = 0,
  marker.only = TRUE,
  scales=list(alternating=FALSE)
)
p_xy
```

![](analysis_files/figure-markdown_github/unnamed-chunk-8-1.png)

Comparison to Flow
------------------

Extract population statistics and merge with flow results:

``` r
# Variant to use for comparison, if there are multiple
variant_name <- 'v00'

# Extract counts for each population in the worfklow as well as associated metadata
df_ck <- getPopStats(subset(gs, variant==variant_name), statistic='freq', format='long') %>%
  mutate(percent=100*Count/ParentCount) %>% 
  filter(str_detect(Population, 'ciCD[4|8][\\+|\\-]ciCD[4|8][\\+|\\-]')) %>%
  rename(population=Population) %>%
  mutate(population=str_replace_all(population, 'ci', '')) %>%
  mutate(donor=str_extract(name, 'D\\d{2}')) %>%
  mutate(replicate=str_extract(name, 'Rep[A|B]')) %>%
  select(donor, population, percent, replicate) %>% 
  as.tibble
  
# Load flow 
df_flow <- read_csv('flow_stats.csv') %>% mutate(replicate='RepA')

df_stats <- bind_rows(
    df_ck %>% mutate(source='cytokit'), 
    df_flow %>% mutate(source='flow')
  ) %>%
  mutate(run=as.character(str_glue('{source}-{replicate}'))) %>%
  filter(population != 'CD4-CD8-') %>% 
  group_by(run, donor, source, replicate) %>% 
  mutate(percent=100*percent/sum(percent)) %>% ungroup %>%
  mutate(pct=str_glue('{p}%', p=round(percent, 0))) 

df_stats %>% head %>% knitr::kable()
```

| donor | population |    percent| replicate | source  | run          | pct |
|:------|:-----------|----------:|:----------|:--------|:-------------|:----|
| D22   | CD4-CD8+   |  20.524691| RepA      | cytokit | cytokit-RepA | 21% |
| D22   | CD4+CD8+   |   5.709877| RepA      | cytokit | cytokit-RepA | 6%  |
| D22   | CD4+CD8-   |  73.765432| RepA      | cytokit | cytokit-RepA | 74% |
| D22   | CD4-CD8+   |  19.833179| RepB      | cytokit | cytokit-RepB | 20% |
| D22   | CD4+CD8+   |   8.248378| RepB      | cytokit | cytokit-RepB | 8%  |
| D22   | CD4+CD8-   |  71.918443| RepB      | cytokit | cytokit-RepB | 72% |

Show percentage comparison as fraction of total (excluding double negative cells since there was no marker in this pan-T cell sample useful for separating these from non-T cells or gamma/delta T cells):

``` r
# Resize + screenshot + save at images/pub/flow_comparison.png
p_stat <- df_stats %>%
  ggplot(aes(x=run, y=percent, fill=population, label=pct)) +
  geom_bar(stat='identity', position='fill', color='white') +
  geom_text(size = 4, position=position_fill(vjust=.7)) +
  scale_fill_brewer(palette='Set1', guide=guide_legend(title='Population')) +
  scale_y_continuous(labels = scales::percent_format()) +
  facet_wrap(~donor, nrow=1) + 
  xlab('') + ylab('') +
  theme_bw() + theme(
    panel.grid.minor.x=element_blank(),
    panel.grid.minor.y=element_blank(),
    panel.grid.major.x=element_blank(),
    panel.grid.major.y=element_blank(),
    axis.ticks.x=element_blank(),
    axis.ticks.y=element_blank(),
    axis.text.y=element_blank(),
    strip.background = element_rect(colour="white", fill="white"),
    axis.text.x = element_text(angle = 90, hjust = 1)
  ) 
p_stat
```

![](analysis_files/figure-markdown_github/unnamed-chunk-11-1.png)

Show scatterplot comparison:

``` r
p_scat <- inner_join(
  df_stats %>% filter(source == 'cytokit') %>% select(donor, population, replicate, percent),
  df_stats %>% filter(source == 'flow') %>% select(donor, population, percent),
  by=c('donor', 'population'),
  suffix=c('.cytokit', '.flow')
) %>% 
  ggplot(aes(x=percent.cytokit, y=percent.flow, color=population, shape=donor)) + 
  geom_point(size=3) + 
  geom_abline(slope=1, linetype='dashed', alpha=.1) + 
  scale_color_brewer(palette='Set1') +
  scale_y_continuous(labels=function(x) scales::percent(x, scale=1)) +
  scale_x_continuous(labels=function(x) scales::percent(x, scale=1)) +
  facet_wrap(~replicate, nrow=1) + 
  labs(x='Cytokit (%)', y='Flow (%)', shape='Donor', color='Population') + 
  theme_bw() + theme(
    panel.grid.minor.x=element_blank(),
    panel.grid.minor.y=element_blank(),
    axis.ticks.x=element_blank(),
    axis.ticks.y=element_blank(),
    strip.background = element_rect(colour="white", fill="white")
  ) 
p_scat
```

![](analysis_files/figure-markdown_github/unnamed-chunk-12-1.png)

Arranged Figures
----------------

For publication:

``` r
gridExtra::grid.arrange(grobs=list(p_xy, p_stat, p_scat), layout_matrix=rbind(c(1, 1), c(2, 3)))
```

![](analysis_files/figure-markdown_github/unnamed-chunk-13-1.png)

Session
-------

``` r
sessionInfo()
```

    ## R version 3.5.1 (2018-07-02)
    ## Platform: x86_64-apple-darwin15.6.0 (64-bit)
    ## Running under: macOS High Sierra 10.13.3
    ## 
    ## Matrix products: default
    ## BLAS: /Library/Frameworks/R.framework/Versions/3.5/Resources/lib/libRblas.0.dylib
    ## LAPACK: /Library/Frameworks/R.framework/Versions/3.5/Resources/lib/libRlapack.dylib
    ## 
    ## locale:
    ## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
    ## 
    ## attached base packages:
    ## [1] parallel  stats     graphics  grDevices utils     datasets  methods  
    ## [8] base     
    ## 
    ## other attached packages:
    ##  [1] bindrcpp_0.2.2            ggcyto_1.10.0            
    ##  [3] openCyto_1.20.0           flowWorkspace_3.30.0     
    ##  [5] ncdfFlow_2.28.0           BH_1.66.0-1              
    ##  [7] RcppArmadillo_0.9.200.4.0 flowCore_1.48.0          
    ##  [9] forcats_0.3.0             stringr_1.3.1            
    ## [11] dplyr_0.7.8               purrr_0.2.5              
    ## [13] readr_1.1.1               tidyr_0.8.2              
    ## [15] tibble_1.4.2              ggplot2_3.1.0            
    ## [17] tidyverse_1.2.1           knitr_1.20               
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] nlme_3.1-137        matrixStats_0.54.0  lubridate_1.7.4    
    ##  [4] RColorBrewer_1.1-2  httr_1.3.1          rprojroot_1.3-2    
    ##  [7] Rgraphviz_2.26.0    tools_3.5.1         backports_1.1.2    
    ## [10] R6_2.3.0            KernSmooth_2.23-15  lazyeval_0.2.1     
    ## [13] BiocGenerics_0.28.0 colorspace_1.3-2    withr_2.1.2        
    ## [16] tidyselect_0.2.5    gridExtra_2.3       mnormt_1.5-5       
    ## [19] compiler_3.5.1      graph_1.60.0        cli_1.0.1          
    ## [22] rvest_0.3.2         Biobase_2.42.0      flowClust_3.20.0   
    ## [25] xml2_1.2.0          labeling_0.3        flowStats_3.40.0   
    ## [28] scales_1.0.0        DEoptimR_1.0-8      hexbin_1.27.2      
    ## [31] mvtnorm_1.0-8       robustbase_0.93-3   RBGL_1.58.0        
    ## [34] digest_0.6.18       rmarkdown_1.10      R.utils_2.7.0      
    ## [37] rrcov_1.4-7         pkgconfig_2.0.2     htmltools_0.3.6    
    ## [40] highr_0.7           rlang_0.3.0.1       readxl_1.1.0       
    ## [43] rstudioapi_0.8      bindr_0.1.1         jsonlite_1.5       
    ## [46] gtools_3.8.1        mclust_5.4.1        R.oo_1.22.0        
    ## [49] magrittr_1.5        Matrix_1.2-15       Rcpp_1.0.0         
    ## [52] munsell_0.5.0       R.methodsS3_1.7.1   stringi_1.2.4      
    ## [55] yaml_2.2.0          MASS_7.3-51.1       zlibbioc_1.28.0    
    ## [58] plyr_1.8.4          grid_3.5.1          crayon_1.3.4       
    ## [61] lattice_0.20-38     splines_3.5.1       haven_1.1.2        
    ## [64] hms_0.4.2           pillar_1.3.0        fda_2.4.8          
    ## [67] corpcor_1.6.9       stats4_3.5.1        XML_3.98-1.16      
    ## [70] glue_1.3.0          evaluate_0.12       latticeExtra_0.6-28
    ## [73] data.table_1.11.8   modelr_0.1.2        cellranger_1.1.0   
    ## [76] gtable_0.2.0        clue_0.3-56         assertthat_0.2.0   
    ## [79] ks_1.11.3           broom_0.5.0         IDPmisc_1.1.18     
    ## [82] pcaPP_1.9-73        flowViz_1.46.0      ellipse_0.4.1      
    ## [85] cluster_2.0.7-1
