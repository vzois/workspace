package sim.testing;

import math.geom2d.Point2D;
import math.geom2d.conic.Circle2D;
import math.geom2d.line.Ray2D;

public class TestMain {

	public static void main(String args[]){
		ray_collision();
	}
	
	public static void ray_collision(){
		Circle2D c = new Circle2D(6,6.5,1.5);
		Ray2D r1 = new Ray2D(new Point2D(2,4.5),new Point2D(8,6.5));
		Ray2D r2 = new Ray2D(new Point2D(5.5,8.5),new Point2D(8,3));
		Ray2D r3 = new Ray2D(new Point2D(6.5,3.5),new Point2D(8,1));
		
		System.out.println("r1"+c.intersections(r1));
		System.out.println("r2"+c.intersections(r2));
		System.out.println("r3"+c.intersections(r3));
	}
	
}
