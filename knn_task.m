% Task 2.4 - Implementation of KNN

% 1: Load the fisheriris.mat dataset
load fisheriris.mat


% 2: Shuffle iris dataset and split data into training and testing data and
% target
rng(1); % set random seed for reproducibility

% Classify the species with the categorical array
cg = categorical(species);
% Sorting and storing species 
dis = categories(cg);
training_data = [];training_target = [];
testing_data = [];testing_target = [];
for i = 1 : length(dis)
    ind = find(cg == dis{i});
    
    % Create random permutation
    ind = ind(randperm(length(ind)));

    %  Dividing data into 60% training and 40% testing
    train_ind = ind(1:round(0.6* length(ind)));
    test_ind = ind(round(0.6* length(ind))+1:end);

    % Creating testing and training dataset with meas
    training_data = [training_data; meas(ind(1:round(length(ind)*0.6)),:)];
    training_target = [training_target; cg(ind(1:round(length(ind)*0.6)),:)];
    testing_data= [testing_data; meas(ind(1+round(length(ind)*0.6):end),:)];
    testing_target = [testing_target; cg(ind(1+round(length(ind)*0.6):end),:)];
end


% 3: Set the K values
k_value = [5,7];

% modelformed=fitcknn(dataTrain,'label');
% Train KNN classifier for fisher's iris dataset
for K = k_value
    modelformed=fitcknn(training_data,training_target,'NumNeighbors',K);


    % 4: Confusion matrix and classification percentage
    % Display the predicted labels
    predicted_group=predict(modelformed,testing_data);

    Confusion_matrix = confusionmat(testing_target, predicted_group);
    % Calculate the accuracy according to the confusion matrix
    accuracy_check = sum(diag(Confusion_matrix)) / sum(Confusion_matrix(:)); 

    % Display the K_value, confusion matrix and percentage of KNN algorithm
    % classification
    fprintf('K_value = %d\n',K);
    disp('Confusion matrix');
    disp(Confusion_matrix);
    fprintf('Percentage of classification %.2f%%\n\n', accuracy_check * 100);

end


% 5. The limitations or drawbacks of KNN

   %•	Finding nearest neighbors for each sample under a large data set is difficult.  Therefore, this increases memory usage.

   %•	Results may be slower when using a large dataset.

   %•	As the size of the dataset increases, the computing cost increases.

   %•	This KNN can be sensitive to irrelevant features.

   %•	Here the performance of the KNN algorithm is reduced due to the high cost of finding the distance between points.  Because of this, KNN does not work easily with large datasets.

   %•	Here the results obtained by not scaling the features may be wrong.



