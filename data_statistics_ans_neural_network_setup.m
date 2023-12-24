% Task 2.1 Data Statistics
% cleaing envirionment
clc ,clear all, close all;

% A
load fisheriris.mat; %Load the dataset

% B
% Calculate the number of samples in the 'meas' matrix
% by determining the size along the first dimension (rows).
N = size(meas, 1);

% C
% % Calculating statistics for each column (feature) from 1 to 4
% % using for loop
for i = 1:4

    
    mean_val = mean(meas(:, i));  % Mean (average) of the data in the current column.
    standard_deviation_val = std(meas(:, i)); %Standard deviation of the data in the current column
    maximum_val = max(meas(:, i)); % Maximum value in the current column.
    minimum_val = min(meas(:, i)); % Minimum value in the current column.
    root_mean_val = rms(meas(:, i)); % Root mean square of the data in the current column.

% % In addtional we are print a all of values(Mean, the Standard Deviation, Maximum,
% % Minimum and Root Mean Square) in command window
    fprintf('Column %d: \n', i);
    fprintf('Mean: %f\n',mean_val);
    fprintf('Standed Deviation is: %f\n', standard_deviation_val);
    fprintf('Maximum Value is: %f\n',maximum_val);
    fprintf('Minimux value is: %f\n', minimum_val);
    fprintf('Root Mean Square is: %f\n\n', root_mean_val);

end

% Task 2.2 Neural Network setup
% 1 ANSWERS
% ASSIGNING THE DATASET TO VARIABLE MEANS_DATASET
means_DataSet = meas;

% ASSIGING NUMARIC VALUE TO SPECIES NAME
species_Type = grp2idx(species);

% JONING/CONCATENATING THE MATRIX
meas_DataSet=[means_DataSet species_Type];

% Randomly shuffle the rows
q= randperm(N);


meas_DataSet = meas_DataSet(q,:);
% training_Data = meas_DataSet(1:90,:);
% testing_Data = meas_DataSet(1:60,:);
training_Data = meas_DataSet(1:round(0.6*N), :);
testing_Data = meas_DataSet(round(0.6*N)+1:end, :);


trainD = []; %  Training Data
trainT =[];  %  Training Target

testD=[];    %  Testing Data
testT=[];    %  Testing Target

% Extracting features (columns 1 to 4) for the training dataset
trainD = training_Data(:, 1:4);

% Extracting the target variable (column 5) for the training dataset
trainT = training_Data(:, 5);

% Extracting features (columns 1 to 4) for the testing dataset
testD = testing_Data(:, 1:4);

% Extracting the target variable (column 5) for the testing dataset
testT = testing_Data(:, 5);

 accuMatrix = zeros(10,4);
 
% 2 AND 3 ANSWERS
% ---------------------------------------------------
hidden_layer_sizes = [10, 15, 20]; %hidden layer sizes 

% repetitions for each hidden layer size
num_repetitions = 4;

% Matrix to store accuracy values for each repetition and hidden layer size
accuMatrix = zeros(num_repetitions, length(hidden_layer_sizes));

% Loop over different hidden layer sizes
for h = 1:length(hidden_layer_sizes)
    hidden_size = hidden_layer_sizes(h); % Extract the current hidden layer size
    
    % Iterate for the specified number of repetitions
    for repeatvalue = 1:num_repetitions
        % Create a feedforward neural network with the current hidden layer size
        net = feedforwardnet(hidden_size);
        
        % Train the neural network
        [net, tr] = train(net, trainD', trainT');
        
        % Test the neural network
        output = net(testD');
        
        % Calculate accuracy
        test_MLPvalue = sum(round(output) == testT') / length(output) * 100;
        
        % Update accuMatrix
        accuMatrix(repeatvalue, h) = test_MLPvalue;
    end
    
    % Calculate and display the mean accuracy for the current hidden layer size
    mean_accuracy = mean(accuMatrix(:, h));
    fprintf('Hidden Layer Size: %d, Mean Accuracy: %.2f%%\n', hidden_size, mean_accuracy);
end

% 4 ANSWER
view(net);% Visualize the trained neural network using the "view" function


% ----------------------------------------------------
% 5 ANSWER

% Define the number of times to test the trained network 
num_oftesting_runs = 10;

% Initialize a variable to store accuracy values for each testing run
testing_accuracies = zeros(1, num_oftesting_runs);

% Perform testing for the specified number of runs
for testing_iteration = 1:num_oftesting_runs
    % Obtain predictions for the testing dataset using the trained neural network
    test_predictions = net(testD');
    
    % Calculate accuracy for this testing run
    current_testing_accuracy = sum(round(test_predictions) == testT') / length(test_predictions) * 100;
    
    % Store accuracy value for this testing run
    testing_accuracies(testing_iteration) = current_testing_accuracy;
end

% Calculate and display the average classifier accuracy over multiple testing runs
average_testing_accuracy = mean(testing_accuracies);
fprintf('Average Classifier Accuracy on Testing Dataset: %.2f%%\n', average_testing_accuracy);


