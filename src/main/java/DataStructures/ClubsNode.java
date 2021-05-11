package DataStructures;

import Utils.Formulas;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/***
 * "The root is associated with the whole data domain."
 * "Each node is associated with a range of the multi-dimensional domain"
 *      - range object, containing upper and lower bounds of each dimension.
 * Each node also maintains summary information about points in its range
 * to expedite the clustering computation (SSQ)
 *      - number of points
 *      - average of points (C_0)
 *      - sum of points (S)
 *      - sum of squares of points (Q)
 */
public class ClubsNode implements Comparable<ClubsNode>{

    // Tree requirements
    INDArray data;
    private ClubsNode leftBlock;
    private ClubsNode rightBlock;
    private ClubsNode parentBlock;
    private ClubsNode siblingBlock;

    // Clubs algorithm data
    private long N;      // # Of points (rows/samples)
    private long d;      // # of dimensions (columns/variables)
    private INDArray ub; // upper bounds
    private INDArray lb; // lower bounds

    private INDArray S;   // Each element of S is the sum of points in a single dimension.
    private INDArray Q;   // Each element of Q is the sum of squares along a single dimension.
    private double SSQ;   // Sum(Q - (S ** 2 / N))


    public ClubsNode(double[][] data) {
        this.data = Nd4j.create(data);
        this.N = this.data.rows();
        this.d = this.data.columns();
        this.ub = this.data.max(0);
        this.lb = this.data.min(0);

        this.S = Nd4j.sum(this.data, 0);
        this.Q = Nd4j.sum(this.data.mul(this.data), 0);
        this.SSQ = Formulas.calcSSQ(S, Q, N);
    }

    @Override
    public String toString(){
        StringBuilder nodeInfo = new StringBuilder("Node information:\n");
        nodeInfo.append(String.format("Number of datapoints: %d\n", N))
                .append(String.format("Lower bounds: %s\n", lb.toString()))
                .append(String.format("Upper bounds: %s\n", ub.toString()))
                .append(String.format("S value: %s\n", S.toString()))
                .append(String.format("Q value: %s\n", Q.toString()))
                .append(String.format("Calculated SSQ: %f\n", SSQ));
        return nodeInfo.toString();
    }

    @Override
    public int compareTo(ClubsNode other) {
        return Double.compare(other.SSQ, this.SSQ);
    }

    public static void main(String[] args) {
        double[][] test = {{1, 2},{3, 4}};

        ClubsNode testNode = new ClubsNode(test);
        System.out.println(testNode);

        INDArray P = Nd4j.create(test);
//        System.out.println("Matrix");
//        System.out.println(P);
//        System.out.println("row 0");
//        System.out.println(P.get(NDArrayIndex.point(0)));
//        System.out.println("sum");
//        System.out.println(Nd4j.sum(P, 0));
//        System.out.println("elementwise squared");
//        System.out.println(P.mul(P));
//        System.out.println("Max of P along dimension 0");
//        System.out.println(P.max(0));
    }
}
