package Utils;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Formulas {
    public static double calcSSQ(INDArray S, INDArray Q, long N) {
        return Q.sub(S.mul(S).div(N)).sum().getDouble(0);
    }
}
