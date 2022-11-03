import pandas
import matplotlib.pyplot as pyplot

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

dataset = pandas.read_csv("dataset.csv")

# print(dataset)

dataset = dataset.values

# print(dataset)



pyplot.scatter(dataset[:,0], dataset[:,1])
pyplot.savefig("scatterplot.png")
pyplot.close()

def run_kmeans(n, dataset):
  machine = KMeans(n_clusters=n)
  machine.fit(dataset)
  results = machine.predict(dataset)
  centroids = machine.cluster_centers_
  ssd = machine.inertia_
  if n > 1:
    silhouette = silhouette_score(dataset, machine.labels_, metric="euclidean")
  else:
    silhouette = 0
  pyplot.scatter(dataset[:,0], dataset[:,1], c=results)
  pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker="*", s = 200)
  pyplot.savefig("scatterplot_kmean_" + str(n) + ".png")
  pyplot.close()
  return ssd, silhouette


result = [run_kmeans(i+1, dataset) for i in range(7)]
# print(result)

ssd_result = [ i[0] for i in result]
pyplot.plot(range(1,8), ssd_result)
pyplot.savefig("kmeans_ssd.png")
pyplot.close()


silhouette_result = [ i[1] for i in result][1:]
pyplot.plot(range(2,8), silhouette_result)
pyplot.savefig("kmeans_silhouette.png")
pyplot.close()

print(silhouette_result)
print(silhouette_result.index(max(silhouette_result))+2)











