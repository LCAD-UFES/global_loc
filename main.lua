require 'torch'
require 'nn'
require 'cutorch'
require 'cudnn'
require 'image'
require 'optim'


local function prepair_samples(csv_name, year, dataset_prefix)

  print("Preparing Samples!\n")
  
  local csv = require("csv")
  local f = csv.open(csv_name)
  local i = -1
  local x, y, px, py, img_file_name
  local num_imgs = 5
  local imgs = torch.Tensor(num_imgs, 3, 224, 224)
  local labels = torch.Tensor(num_imgs)
  for fields in f:lines() do
    if i ~= -1 then
      local label
      for j, v in ipairs(fields) do
        if j == 1 then img_file_name = year .. v .. ".bb08.l.png" end 
        if j == 2 then x = tonumber(v) end 
        if j == 3 then y = tonumber(v) end 
        if j == 4 then label = tonumber(v) end 
      end
      if i >= 1 and i <= num_imgs then
        print(img_file_name, ' dist = ', math.sqrt((x - px) * (x - px) + (y - py) * (y - py)))
        local img = image.load(img_file_name, 3, 'byte')
        img = image.scale(img, 224, 224, 'bilinear')
        imgs[i] = img
        labels[i] = label
        image.display{image = imgs[i], zoom = 1, legend = 'image ' .. tostring(i)}
      end
      px = x
      py = y
    end
    i = i + 1
  end
--  print(imgs:size())
--  print(labels:size())
--  for i = 1, labels:size(1), 1 do
--    print(labels[i])
--  end
  
  model = torch.load('/home/alberto/neuraltalk2/cnn.model')
  model = model:cuda()
  -- print(model)
  model:evaluate()
  -- local images = torch.load('/home/alberto/neuraltalk2/image.data')
  local images = imgs
  print(images:size())
  images = images:cuda()
  local cnn_output = model:forward(images)
  torch.save(dataset_prefix .. "_images.tensor", images)
  torch.save(dataset_prefix .. "_labels.tensor", labels)
  torch.save(dataset_prefix .. "_cnn_out.tensor", cnn_output)
  
  for i = 1, images:size(1), 1 do
    local cnn_output_ret = torch.reshape(cnn_output[i], 24, 32)
    --local cnn_output_image = image.scale(cnn_output_ret:double(), 32*8, 24*8, 'simple')
    
    image.display{image = images[i], zoom = 1, legend = 'image ' .. tostring(i)}
    image.display{image = cnn_output_ret, zoom = 8, legend = 'model ' .. tostring(i)}
  end
end


local function build_net(cnn_output, labels)
  -- define model to train
  model = nn.Sequential()
  model:add(nn.Linear(cnn_output:size(2), 200))
  model:add(nn.ReLU())
  model:add(nn.Linear(200, labels:max()))
  model:add(nn.LogSoftMax())
  
  -- a negative log-likelihood criterion for multi-class classification
  criterion = nn.ClassNLLCriterion()
end


local function train_net(batchInputs, batchLabels)
-- https://github.com/torch/nn/blob/master/doc/training.md
  local optimState = {learningRate = 0.05, learningRateDecay = 5e-7}
  --model = model:cuda()
  --batchInputs = batchInputs:cuda();
  --batchLabels = batchLabels:cuda();
  --criterion = criterion:cuda();
  
  local params, gradParams = model:getParameters()
  
  for epoch=1,2000 do
    -- local function we give to optim
    -- it takes current weights as input, and outputs the loss
    -- and the gradient of the loss with respect to the weights
    -- gradParams is calculated implicitly by calling 'backward',
    -- because the model's weight and bias gradient tensors
    -- are simply views onto gradParams
    local function feval(params)
      gradParams:zero()
      local outputs = model:forward(batchInputs)
      --outputs = outputs:cuda();
      local loss = criterion:forward(outputs, batchLabels)
      local dloss_doutput = criterion:backward(outputs, batchLabels)
      --dloss_doutput = dloss_doutput:cuda()
      model:backward(batchInputs, dloss_doutput)
      return loss,gradParams
    end
    --feval(params)
    optim.sgd(feval, params, optimState)
    --print("epoch = ", epoch)
  end
end


local function test_net(cnn_output, labels)
  local outputs = model:forward(cnn_output)
  print(labels)
  print(outputs:exp())
end


local function main()
  --prepair_samples("/dados/GPS_clean/UFES-2012-30-train.csv", "/dados/GPS_clean/2012/", "training") -- depois que preparar, nao precisa rodar de novo
  --prepair_samples("/dados/GPS_clean/UFES-2014-30-train.csv", "/dados/GPS_clean/2014/", "test") -- depois que preparar, nao precisa rodar de novo
  local images = torch.load("training_images.tensor")
  local cnn_output = torch.load("training_cnn_out.tensor")
  local labels = torch.load("training_labels.tensor")
  
  print("Build net")
  build_net(cnn_output, labels)
  -- print(model)

  -- treinar a rede com cada cnn_output (input) com o label correspondente (target)
  print("Train net")
  train_net(cnn_output:double(), labels)

  -- testar com as images de 2012 (teste de sanidade) e depois com images de 2014 (tem que ler e gerar tensor em forma de arquivo)
  print("Test net")
  local images = torch.load("test_images.tensor")
  local cnn_output = torch.load("test_cnn_out.tensor")
  local labels = torch.load("test_labels.tensor")
  test_net(cnn_output:double(), labels) -- os mesmos <images, labels> (sanidade) ou os de 2014
  
end

main()
