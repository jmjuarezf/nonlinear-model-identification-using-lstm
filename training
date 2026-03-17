clear
close all
clc
% Fijar semilla para reproducibilidad
rng(42);

load dataset_entreno.mat %este conjunto de datos lo uso para entrenamiento
[datasetn,C,S]=normalize(dataset);
save C C; %guardar media y desviación típica para poder normalizar más datos
save S S;

%los datos de entrada y salida se introducen por vector columna. Cada
%vector columna pertenece a una variable (o campo de entrada en
%terminología de bases de datos)

XTrain=[datasetn(:,1) datasetn(:,2);]; %entradas p (bomba) y v (válvula)
TTrain=[datasetn(:,3) datasetn(:,4)]; %salidas h1 (altura depósito 1)


load dataset_validation.mat %este conjunto de datos lo uso para validación
% [dataset_validation_n,C,S]=normalize(dataset);
dataset_validation_n = normalize(dataset, "center", C, "scale", S);
XValidation=[dataset_validation_n(:,1) dataset_validation_n(:,2);]; %entradas p (bomba) y v (válvula)
TValidation=[dataset_validation_n(:,3) dataset_validation_n(:,4)]; %salidas h1 (altura depósito 1)


%Definir la arquitectura de red
%Defina la arquitectura de la red. Cree una red de LSTM que conste de una capa de LSTM con 
%NumHiddenUnits como número de unidades ocultas , seguida de una capa totalmente conectada 
%de tamaño 50 y una capa de abandono con probabilidad de abandono del 0,5.
numFeatures=size(XTrain,2);
numResponses = size(TTrain,2);
numHiddenUnits = 150; %Tenía 150


layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,OutputMode="sequence")
    batchNormalizationLayer  % Normalización por lotes
    fullyConnectedLayer(numResponses)];
%Especifique las opciones de entrenamiento. Entrene durante 60 épocas con minilotes de tamaño 20 mediante el solver "adam". Especifique una tasa de aprendizaje de 0,01. Para evitar que los gradientes exploten, establezca el umbral del gradiente en 1. Para mantener las secuencias ordenadas por su longitud, establezca la opción Shuffle en "never". Muestre el progreso del entrenamiento en una gráfica y monitorice la métrica de error cuadrático medio raíz (RMSE).

maxEpochs = 1500; %800
miniBatchSize = 12; %12

options = trainingOptions("adam", ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    InitialLearnRate=0.001, ... %.0.001
    GradientThreshold=1, ...
    Shuffle="never", ...    
    Metrics="rmse", ...
    ValidationData={XValidation,TValidation}, ...
    ValidationFrequency=10, ...
    OutputNetwork="best-validation", ...
    Plots="training-progress", ...
    Verbose=0);


%Entrenar la red
%Entrene la red neuronal con la función trainnet. Para la regresión, utilice la pérdida de error cuadrático medio. De forma predeterminada, la función trainnet usa una GPU en caso de que esté disponible. Para entrenar en una GPU se requiere una licencia de Parallel Computing Toolbox™ y un dispositivo GPU compatible. Para obtener información sobre los dispositivos compatibles, consulte GPU Computing Requirements (Parallel Computing Toolbox). De lo contrario, la función trainnet usa la CPU. Para especificar el entorno de ejecución, utilice la opción de entrenamiento ExecutionEnvironment.

[net,info] = trainnet(XTrain,TTrain,layers,"mse",options);

save net net

% testm
