# global_loc

Para rodar treinando e testando com os dados de treino (teste de sanidade) e usando arquivos pré-processados:
 qlua main.lua -e 200 -m 0.0 -s
 
Para rodar treinando e testando com dados distintos e usando arquivos pré-processados:
 qlua main.lua -e 200 -m 0.0
 
Para rodar treinando e testando com dados distintos e pré-processando 12 amostras de dado (requer dados conforme indicado no fonte, para ver as imagens adicione -d no fim da linha abaixo):
 qlua main.lua -e 200 -m 0.0 -p -n 12
 