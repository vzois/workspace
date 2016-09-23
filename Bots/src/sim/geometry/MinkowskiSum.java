package sim.geometry;

import java.util.ArrayList;

import math.geom2d.Point2D;
import math.geom2d.polygon.SimplePolygon2D;
import math.geom2d.polygon.convhull.JarvisMarch2D;

public class MinkowskiSum {
	private SimplePolygon2D A,B;
	private ArrayList<Point2D> points;
	
	public MinkowskiSum(SimplePolygon2D A,SimplePolygon2D B){
		this.A = A;
		this.B = B;
	}
	
	public void sum(){
		ArrayList<Point2D> aList = new ArrayList<Point2D>(A.vertices());
		ArrayList<Point2D> bList = new ArrayList<Point2D>(B.vertices());
		this.points = new ArrayList<Point2D>(aList.size()*bList.size());
		
		for(Point2D aPoint : aList){
			for(Point2D bPoint : bList){
				if(!aPoint.contains(bPoint)){
					points.add(aPoint.plus(bPoint));
				}
			}
		}
	}
	
	public SimplePolygon2D getConvexHull(){
		return (SimplePolygon2D) (new JarvisMarch2D()).convexHull(points);
	}
}
