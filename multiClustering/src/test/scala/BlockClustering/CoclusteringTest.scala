package BlockClustering

import Common.Tools._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{Gamma, RandBasis}

import scala.util.Random
import scala.collection.mutable.ListBuffer
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import breeze.storage.Zero
import org.scalatest.funsuite.AnyFunSuite

class CoclusteringTest extends AnyFunSuite {

  val modes = List(List(DenseVector(1.9, 1.9), DenseVector(1.4, -1.4)),
    List(DenseVector(-1.7, 1.7),DenseVector(-1.7, -1.7)),
    List(DenseVector(-3, -3D),DenseVector(5, 5D)))

  val covariances = List(
    List(DenseMatrix(0.25, 0.2, 0.2, 0.25).reshape(2,2),
      DenseMatrix(0.17, 0.1, 0.1, 0.17).reshape(2,2)),
    List(DenseMatrix(0.12, 0.0, 0.0, 0.12).reshape(2,2),
      DenseMatrix(0.3, 0.0, 0.0, 0.3).reshape(2,2)),
    List(DenseMatrix(0.22, 0.0, 0.0, 0.15).reshape(2,2),
      DenseMatrix(0.1, 0.0, 0.0, 0.1).reshape(2,2)))

  val Data: List[List[DenseVector[Double]]] = matrixToDataByCol(Common.DataGeneration.randomLBMDataGeneration(
    modesByCol = modes,
    covariancesByCol = covariances,
    sizeClusterRow = List(30, 30),
    sizeClusterCol = List(15, 15, 15))).transpose
  val n = 60
  val p = 45

  implicit val optDenseVectZero: Zero[DenseVector[Double]] = Zero(DenseVector(0.0))
  implicit val randbasis: RandBasis = RandBasis.withSeed(54321)


  test("Default constructor") {
    {

      val coclustering = new Coclustering(Data, alphaRow = Some(5D), alphaCol = Some(10D))
      assert(coclustering.rowClustering.n == 60)
      assert(coclustering.rowClustering.p == 45)
      assert(coclustering.rowClustering.d == 2)
      assert(coclustering.rowClustering.colPartition == (0 until 45).toList)
      assert(coclustering.rowClustering.rowPartition == List.fill(60)(0))
      assert(coclustering.rowClustering.countColCluster == ListBuffer.fill(45)(1))
      assert(coclustering.rowClustering.countRowCluster == ListBuffer(60))
      assert(coclustering.rowClustering.NIWParamsByRow.length == 1 & coclustering.rowClustering.NIWParamsByRow.head.length == 45)

      assert(coclustering.rowClustering.NIWParamsByRow.head.head.checkNIWParameterEquals(coclustering.rowClustering.prior.update(Data.map(_.head))))

      assert(! coclustering.rowClustering.updateAlphaFlag)
      assert(coclustering.rowClustering.actualAlphaPrior.scale == 1D & coclustering.rowClustering.actualAlphaPrior.shape == 1D)
      assert(coclustering.rowClustering.actualAlpha == 5D)

      assert(coclustering.colClustering.p == 60)
      assert(coclustering.colClustering.n == 45)
      assert(coclustering.colClustering.d == 2)

      assert(coclustering.colClustering.rowPartition == (0 until 45).toList)

      assert(coclustering.colClustering.colPartition == List.fill(60)(0))
      assert(coclustering.colClustering.countRowCluster == ListBuffer.fill(45)(1))
      assert(coclustering.colClustering.countColCluster == ListBuffer(60))

      assert(coclustering.colClustering.NIWParamsByRow.head.head.checkNIWParameterEquals(coclustering.colClustering.prior.update(Data.map(_.head))))
      assert(! coclustering.colClustering.updateAlphaFlag)
      assert(coclustering.colClustering.actualAlphaPrior.scale == 1D & coclustering.colClustering.actualAlphaPrior.shape == 1D)
      assert(coclustering.colClustering.actualAlpha == 10D)

      assert(coclustering.rowClustering.NIWParamsByRow == coclustering.colClustering.NIWParamsByRow.transpose)

    }
  }

  test("Auxliary constructor") {
    {
      val initRowPartition = Random.shuffle((0 until n).toList)
      val initColPartition = Random.shuffle((0 until p).toList)

      val coclustering = new Coclustering(Data,
        alphaRowPrior = Some(Gamma(5D, 10D)(randbasis)),
        alphaColPrior = Some(Gamma(3D, 6D)(randbasis)),
        initByUserRowPartition = Some(initRowPartition),
        initByUserColPartition = Some(initColPartition))

      assert(coclustering.rowClustering.actualAlphaPrior == Gamma(5D, 10D)(randbasis))
      assert(coclustering.rowClustering.actualAlpha == 50D)
      assert(coclustering.rowClustering.rowPartition == initRowPartition)
      assert(coclustering.rowClustering.colPartition == initColPartition)
      assert(coclustering.rowClustering.NIWParamsByRow.length == 60)
      assert(coclustering.rowClustering.NIWParamsByRow.head.length == 45)

      assert(coclustering.colClustering.actualAlphaPrior == Gamma(3D, 6D)(randbasis))
      assert(coclustering.colClustering.actualAlpha == 18D)
      assert(coclustering.colClustering.rowPartition == initColPartition)
      assert(coclustering.colClustering.colPartition == initRowPartition)
      assert(coclustering.colClustering.NIWParamsByRow.length == 45)
      assert(coclustering.colClustering.NIWParamsByRow.head.length == 60)

      assert(coclustering.rowClustering.NIWParamsByRow == coclustering.colClustering.NIWParamsByRow.transpose)

    }
  }
}
