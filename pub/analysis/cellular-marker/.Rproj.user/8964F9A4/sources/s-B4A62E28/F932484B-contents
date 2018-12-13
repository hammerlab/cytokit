library(tidyverse)
library(flowCore)
library(ggcyto)

#filter <- flowCore::filter

experiments <- c(
  '20180614_D22_RepA_Tcell_CD4-CD8-DAPI_5by5',
  '20180614_D22_RepB_Tcell_CD4-CD8-DAPI_5by5',
  '20180614_D23_RepA_Tcell_CD4-CD8-DAPI_5by5',
  '20180614_D23_RepB_Tcell_CD4-CD8-DAPI_5by5'
)
variants <- c('v00', 'v01', 'v02', 'v03')
gsm <- expand.grid(experiments, variants) %>% set_names(c('experiment', 'variant')) %>%
  mutate(path=str_glue('/lab/data/{experiment}/output/{variant}/cytometry/data.fcs')) %>%
  mutate(donor=str_extract(experiment, 'D\\d{2}')) %>%
  mutate(replicate=str_extract(experiment, 'Rep[A|B]')) %>%
  mutate(sample=str_glue('{donor}_{replicate}_{variant}')) %>%
  as('AnnotatedDataFrame')
sampleNames(gsm) <- gsm@data$sample

load_fcs <- function(path, donor, replicate) {
  fr <- read.FCS(path)
  if (donor == 'D22' && replicate == 'RepB'){
    d <- exprs(fr)
    mask <- d[,'tilex'] < 4 | d[,'tiley'] > 2
    print(sprintf('Removing %s rows of %s for file %s', sum(!mask), nrow(fr), path))
    fr <- Subset(fr, mask)
  }
  if (donor == 'D23' && replicate == 'RepB'){
    d <- exprs(fr)
    mask <- (d[,'tilex'] != 1 | d[,'tiley'] != 1) & (d[,'tilex'] != 1 | d[,'tiley'] != 2)
    print(sprintf('Removing %s rows of %s for file %s', sum(!mask), nrow(fr), path))
    fr <- Subset(fr, mask)
  }
  fr
}
fsr <- gsm@data %>% select(path, donor, replicate) %>% 
  pmap(load_fcs) %>% set_names(gsm@data$sample)

fsr <- flowSet(fsr)
sampleNames(fsr) <- gsm@data$sample
phenoData(fsr) <- gsm


chnl <- c("ciCD4", "ciCD8")
trans <- transformList(chnl, biexponentialTransform())
fst <- transform(fsr, trans)

gs <- GatingSet(fst)
markernames(gs) <- fsr@colnames %>% set_names(fsr@colnames)


#### Gating (v1)

# add(gs, rectangleGate("ciDAPI"=c(50, 150), filterId='dapi'), parent='root')
# add(gs, rectangleGate("celldiameter"=c(3, 15), filterId='celldiameter'), parent='dapi')
# add(gs, rectangleGate("nucleusdiameter"=c(3, 15), filterId='nucleusdiameter'), parent='celldiameter')
# add(gs, rectangleGate("cellcircularity"=c(80, Inf), filterId='cellcircularity'), parent='nucleusdiameter')
# add(gs, rectangleGate("nucleuscircularity"=c(80, Inf), filterId='nucleuscircularity'), parent='cellcircularity')
# add(gs, rectangleGate("cgnneighbors"=c(-Inf, 3.5), filterId='neighbors'), parent='nucleuscircularity')
# 
# 
# for (i in 1:length(gs)){
#   gh <- gs[[i]]
#   fr <- getData(gh)
#   g1 <- mindensity2(fr, channel = 'ciCD4', filterId='CD4+', gate_range=c(3.5, 6))
#   g2 <- mindensity2(fr, channel = 'ciCD8', filterId='CD8+', gate_range=c(4, 5.5))
#   g <- quadGate(ciCD4=g1@min, ciCD8=g2@min)
#   add(gh, g, parent='neighbors', names=c('CD4-CD8+', 'CD4+CD8+', 'CD4+CD8-', 'CD4-CD8-'))
# }
# recompute(gs)
# 
# getPopStats(gs, format='wide')

#### Gating (v2)

add_pop(
  gs, alias="dapi", pop="+", parent='root',
  dims='niDAPI,ciDAPI', gating_method='flowClust.2d',
  gating_args="K=1"
)
add_pop(
  gs, alias="circularity", pop="+", parent='dapi',
  dims='cellcircularity,nucleuscircularity', gating_method='flowClust.2d',
  gating_args='K=1'
)
add_pop(
  gs, alias="diameter", pop="+", parent='circularity',
  dims='celldiameter,nucleusdiameter', gating_method='flowClust.2d',
  gating_args='K=1'
)
# add_pop(
#   gs, alias="*", pop="+/-+/-", parent='diameter',
#   dims='ciCD4,ciCD8', gating_method='mindensity2',
#   gating_args="gate_range=c(3,5.5)"
# )
add_pop(
  gs, alias="*", pop="*", parent='diameter',
  dims='ciCD4,ciCD8', gating_method='quadGate.tmix',
  gating_args="K=3"
)


#### Plots
getGate(gs, 'ciCD4-ciCD8-')

getNodes(gs)
plot(gs)

# Visualize single gate across multiple samples
plotGate(gs, 'dapi')
plotGate(gs, 'diameter')
plotGate(gs, 'circularity')

# Visualize multiple gates for one sample
plotGate(gs[[1]], 'dapi', path='auto', default.x='ciDAPI', default.y='celldiameter')
plotGate(gs[[2]], default.y='celldiameter') # All gates

plotGate(gs, getNodes(gs, path=1)[7:10], xlim=c(1, 7), ylim=c(1, 7), gpar=list(nrow=4, ncol=4)) 
plotGate(gs, getNodes(gs, path=1)[7:10], gpar=list(nrow=4, ncol=4, scales='free')) 

# Visualize 2D expression across multiple samples
autoplot(getData(gs), x='celldiameter', y='nucleusdiameter', bins=50)
autoplot(getData(gs), x='niDAPI', y='ciDAPI', bins=50)

# Pass gates to visualization
all.gates <- getGate(gs, 'ciCD4-ciCD8-')
gates <- sapply(sampleNames(gs), function(sn) all.gates[[sn]])

# ggcyto example
ggcyto(gs[[1]], aes(x='ciCD4', y='ciCD8')) + geom_hex(bins=100) + 
  geom_gate(getNodes(gs, path=1)[6:9]) + geom_stats()


### Final Plots

get_density <- function(x, y){
  d <- densCols(x, y, colramp=colorRampPalette(c("black", "white")))
  as.numeric(col2rgb(d)[1,] + 1L)
}

extract_df <- function(i){
  gh <- gs[[i]]
  pd <- pData(gh)
  gates <- getGate(gh, 'CD4+CD8+')@min
  gs[[i]] %>% getData %>% exprs %>% as_tibble %>%
    mutate(
      sample=as.character(pd$sample), donor=pd$donor, 
      replicate=pd$replicate, variant=pd$variant,
      gCD4=gates['ciCD4'], gCD8=gates['ciCD8']
    ) %>% ungroup
}
variant_name <- 'v02'
df <- map(1:length(gs), extract_df) %>% bind_rows %>%
  group_by(sample) %>% mutate(density=get_density(ciCD4, ciCD8)) %>%
  filter(variant==variant_name)

getPopStats(subset(gs, variant==variant_name), format='wide')

# Resize + screenshot + save at images/pub/cd4_vs_cd8.png
p_xy <- df %>%
  ggplot(aes(x=ciCD4, y=ciCD8)) + 
  geom_point(aes(color=density), size=.1, alpha=.5) + 
  scale_color_distiller(palette='Spectral', direction=-1) +
  geom_vline(data=df %>% group_by(sample) %>% summarize(v=gCD4[1]), aes(xintercept=v)) +
  geom_hline(data=df %>% group_by(sample) %>% summarize(v=gCD8[1]), aes(yintercept=v)) +
  facet_wrap(~sample, scales='free', nrow=2) + 
  guides(color=FALSE) +
  theme_bw() + theme(
    panel.grid.minor.x=element_blank(),
    panel.grid.minor.y=element_blank(),
    panel.grid.major.x=element_blank(),
    panel.grid.major.y=element_blank(),
    axis.ticks.x=element_blank(),
    axis.ticks.y=element_blank(),
    axis.text.x=element_blank(),
    axis.text.y=element_blank(),
    
    # Labels versions
    strip.background = element_rect(colour="white", fill="white")
    
    # Blank version
    # strip.text.x = element_blank()
  )
p_xy


#### Flow Comparison

df_ck <- getPopStats(subset(gs, variant==variant_name), statistic='freq', format='long') %>%
  mutate(percent=100*Count/ParentCount) %>% filter(str_detect(Population, 'CD')) %>%
  rename(population=Population) %>%
  mutate(donor=str_extract(name, 'D\\d{2}')) %>%
  mutate(replicate=str_extract(name, 'Rep[A|B]')) %>%
  select(donor, population, percent, replicate)
  
df_flow <- tribble(
  ~donor, ~population, ~percent,
  'D22', 'CD4+CD8+', .68,
  'D22', 'CD4+CD8-', 75.8,
  'D22', 'CD4-CD8+', 17.5,
  'D22', 'CD4-CD8-', 6.01,
  'D23', 'CD4+CD8+', 2.03,
  'D23', 'CD4+CD8-', 54.1,
  'D23', 'CD4-CD8+', 33.3,
  'D23', 'CD4-CD8-', 10.5
) %>% mutate(replicate='RepA')

df_stats <- df_ck %>% inner_join(df_flow, by = c('donor', 'population')) 

df_stats %>%
  ggplot(aes(x=percent.x, y=percent.y, color=population)) +
  geom_point() + 
  facet_grid(replicate.x~donor)

# Resize + screenshot + save at images/pub/flow_comparison.png
p_stat <- bind_rows(df_ck %>% mutate(source='cytokit'), df_flow %>% mutate(source='flow')) %>%
  mutate(source=str_glue('{source}-{replicate}')) %>%
  mutate(pct=str_glue('{p}%', p=round(percent, 0))) %>%
  filter(population != 'CD4-CD8-') %>%
  ggplot(aes(x=source, y=percent, fill=population, label=pct)) +
  geom_bar(stat='identity', position='fill', color='white') +
  #geom_text(size = 4, position=position_fill(vjust=.5)) +
  scale_fill_brewer(palette='Set1', guide=guide_legend(title='')) +
  scale_y_continuous(labels = scales::percent_format()) +
  facet_wrap(~donor) + 
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
    
    # Labels version
    axis.text.x = element_text(angle = 90, hjust = 1),
    
    # Blank version
    # axis.text.x = element_blank(),
    # strip.text.x = element_blank()
  ) 
p_stat

gridExtra::marrangeGrob(list(p_xy, p_stat), nrow=1, ncol=2)

#### Graveyard

# add(gs, 
#   quadGate(filterId='celltype', list("ciCD4"=4.5, "ciCD8"=5)), 
#   parent='neighbors', names=c('CD4+CD8-', 'CD4+CD8+', 'CD4-CD8-', 'CD4-CD8+')
# )

# .quadrGate <- function(fr, pp_res, channels=NA, filterId="", ...){
#   print(list(...))
#   res <- flowStats::quadrantGate(fr, channels, filterId=filterId, refLine.1 = 5, refLine.2 = 5)
#   return(res)
# }
# registerPlugins(fun=.quadrGate, methodName='quadrGate')

# openCyto::add_pop(
#   gs, alias="*", pop="-/++/-", parent='root',
#   dims='ciCD4,ciCD8', gating_method='mindensity2',
#   gating_args='gating_range=c(3.5, 5)'
# )
# recompute(gs)

# for (node in getNodes(gs, path=1)[6:9]) Rm(node, gs)
# for (node in getNodes(gs, path=1)[6:7]) Rm(node, gs)

# getNodes(gs)
