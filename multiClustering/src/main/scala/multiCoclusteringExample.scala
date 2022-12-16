import BlockClustering.MultiCoclustering
import breeze.linalg.{DenseVector}
import breeze.stats.distributions.{Gamma, RandBasis, ThreadLocalRandomGenerator}
import org.apache.commons.math3.random.MersenneTwister
import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization.writePretty
import org.json4s.native.JsonMethods

import java.io.{File, PrintWriter}
import scala.io.Source

object multiCoclusteringExample {

  def main(args: Array[String]) {

    val nItems: Int = 1000
    val dimPCA: Int = 3

    val alphaShape = 10
    val alphaScale = 1

    val betaShape = 10
    val betaScale = 5

    val gammaShape = 10
    val gammaScale = 5

    val t0 = System.nanoTime()

    val Folder: String = s"../preprocessing/dimensionalityReduction/reducedData/pca$dimPCA/"

    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(42)))

    implicit val formats = DefaultFormats
    val data : List[List[DenseVector[Double]]] = List.range(1,nItems+1).map(x => JsonMethods.parse(Source.fromFile( Folder++s"/$x.json").reader()).children.map(y=>DenseVector(y.children.map(z=>z.extract[Double]):_*)))
    println(data.length)

    val alphaPrior = Gamma(shape = alphaShape, scale = alphaScale )
    val betaPrior = Gamma(shape = betaShape , scale = betaScale )
    val gammaPrior = Gamma(shape = gammaShape ,scale = gammaScale )

    val MCC = new MultiCoclustering(data, alphaPrior, betaPrior, gammaPrior)

    val (varPartition, rowPartitions) = MCC.run(30, verbose = true)

    val t1 = System.nanoTime()

    val results = writePretty(Map("Total processing time"-> (t1-t0)/1e9D,
      "dimPCA" -> dimPCA,
      "varPartition"  -> varPartition,
      "rowPartitions" -> rowPartitions,
      ))
    println(results)
    val f = new File(s"../../finalPartition"+s"_$dimPCA"+".json")
    val w = new PrintWriter(f)
    w.write(results)
    w.close()

  }

}
