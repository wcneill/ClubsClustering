package Utils;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MatrixOps {

    public static INDArray sum(INDArray matrix, int dimension) {
        INDArray cumsum = matrix.cumsum(dimension);
        return cumsum.getRow(dimension);
    }
}
