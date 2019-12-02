# Self Organizing Map. Part 1. Implementing SOM from scratch (initializer, update rules, grid size)

- concept
  - structure : grid, learning steps, input space -> output space mapper
  - competitive vs corperative learning
  - update rules : stochastic gradient with small learning rate
  - goal : to visualize high dimensional space with 2D map (so they generally use 2D grid)
- vanilla version
  - use unit grid
  - masks
- improve update rules
  - The grid points learned from stochastic gradient with small learning rate is oriented to the mean of converged clusters
  - minibatch gradient
  - masks vs iter
  - adjust stuck points : neighbors of neighbors should be simiar. If my neighbors are moved, I also move closer to them (inverse neighbor index like NN-descent)
- initializer
  - random, pca, rp, close to zero, out-ranged vs in-ranged
- experiences
  - soydata

# Self Organizing Map. Part 2. Visualizing High-dimensional Space and Topic-SOM

- initializing grid with sparse matrix (selection based)
- project high dimensional space to 2D grid map
- learning empty space (add constraint to grid)
  - density estimation: outlier detection
- Documents to SOM (keyword lookup)
- LDA to SOM (Hover tool)
- evaluation
  - quantization error : distance from BMU
  - topological error : percentage of BMU and second BMU are not adjacent units

# Self Organizing Map. Part 3. Re-implementing SOM with PyTorch

- available to use GPU
- comparison of training time
- reference : https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html

# Others : Visualizing trajectories

- Encode high dimensional space as SOM grid coordinates and track it
