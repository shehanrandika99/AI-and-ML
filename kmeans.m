% Clear Command Window,Clear Workspace Variables,
%
clc,clear all,close all;;

% QUESTION 1
load('kmeansdata.mat'); %Load The Dataset

% ---------------------------------------------------------------------
% QUESTION 2
% set the range of K_values that values are 3,4,5
k_values = 3:5;

% Initialize an array to the store in silhouette socres
cluster_scores = zeros(size(k_values));

% Iterate through different values of k
for i = 1:length(k_values)
    k = k_values(i); 
    
     % Perform of K-means clustering
    [cluster_indices, centroids] = kmeans(X, k); 
    % Compute silhouette scores
    s = silhouette(X, cluster_indices);
    
    % Calculate of mean silhouette score for the current k value
    cluster_scores(i) = mean(s);
    
    figure; %plot in the silhouette score
    silhouette(X, cluster_indices);
    title(['Silhouette Plot for K = ' num2str(k)]);
end

% -------------------------------------------------------------
% QUESTION 3
% PRINT the main silhouette score fpr each values of K
fprintf('Mean silhouette scores for each K:\n');
for i = 1:length(k_values)
    fprintf('K = %d: %.3f\n', k_values(i), cluster_scores(i));
end


% find the index of the maxsimum mean silhouette score
[~,max_index] = max(cluster_scores);

% print the best number of clusters
fprintf('The best number of clusters %d\n', k_values(max_index))
% plot the mean silhouette scores as a function of K
plot(k_values,cluster_scores,'-o')
xlabel('Number of clusters (k=3,4 and 5)')
ylabel('Mean silhouette score')


% -------------------------------------------------------------------------
% QUESTION 4

% Set the range of k values to evaluate
k_values = 3:5;
% Loop over the different values of k
for k = k_values
    % PERFORM OF THE K-MEANS CLUSTERING WITH T CURRENT  K VALUE
    [cluster_indices, centroids] = kmeans(X, k);

    % PLOT OF THE DATA POINTS WITH DIFFERENT COLORS FOR EACH CLUSTER
    figure;
    gscatter(X(:,1), X(:,2), cluster_indices, 'rgbcmy', 'o', 7)

    % PLOT THE CLUSTER CENTROIDS
    hold on;
    plot(centroids(:,1), centroids(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3)

    title(sprintf('kmeans clustering for k = %d', k));

    % ADD LEGEND
    legend_strings = cell(1, k);
    for i = 1:k
        legend_strings{i} = sprintf('Cluster %d', i);
    end
    legend(legend_strings); % Add legend to the plot using the generated labels

    % ADD AXIS LABELS
    xlabel('X Value');  % X Axis
    ylabel('Y Value');  % Y Axis


    % LABEL CENTROIDS WITH X-COORDINATES
    for i = 1:size(centroids, 1)
        text(centroids(i, 1), centroids(i, 2), sprintf('X: %.2f', centroids(i, 1)), 'Color', 'black', 'FontSize', 8)
    end

    hold off;
end
% ------------------------------------------------------------
% QUESTION 5
% THE BEST NUMBER OF CLUSTER IS K=4 (MORE DETAILS EXPLAIN IT IN DOCUMENT)