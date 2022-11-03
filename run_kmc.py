import pandas
import matplotlib.pyplot as pyplot


from sklearn.cluster import KMeans

dataset = pandas.read_csv("dataset.csv")

# print(dataset)

dataset = dataset.values

# print(dataset)



pyplot.scatter(dataset[:,0], dataset[:,1])
pyplot.savefig("scatterplot.png")
pyplot.close()





kmeans_machine = KMeans(n_clusters=4)
kmeans_machine.fit(dataset)
results = kmeans_machine.predict(dataset)
pyplot.scatter(dataset[:,0], dataset[:,1], c=results)
pyplot.savefig("scatterplot_kmean_4.png")
pyplot.close()

