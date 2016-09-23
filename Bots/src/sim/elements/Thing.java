package sim.elements;

import java.awt.Color;
import java.awt.geom.Ellipse2D;
import java.util.Random;

import sim.structures.Vector;

public class Thing implements Comparable<Thing>,Cloneable {
	protected double size;
	protected Color color;
	protected Vector loc;
	protected Vector velocity;
	protected Vector sortBy;
	
	public Thing(double size){
		this(size,Color.WHITE);
	}
	
	public Thing(double size, Color color){
		this.size = size;
		this.color = color;
		this.velocity = new Vector();
	}
	
	public Color getC(){
		return this.color;
	}
	
	public void setC(Color color){
		this.color = color;
	}
	
	public double getS(){
		return this.size;
	}
	
	public void setL(double x,double y){
		this.loc = new Vector(x,y);
	}
	
	public void setL(Vector pos){
		this.loc = new Vector(pos);
	}
	
	public Vector getL(){
		return this.loc;
	}
	
	public Vector getV(){// GET VELOCITY
		return this.velocity;
	}
	
	public void setRL(int maxX,int minX,int maxY,int minY, Random seed){// SET BOT TO RANDOM LOCATION//
		this.loc = Vector.rand(minX+(int)Math.ceil(this.size), maxX-(int)Math.ceil(this.size)+1, minY+(int)Math.ceil(this.size), maxY-(int)Math.ceil(this.size)+1, seed);		
	}

	public void set(Thing sortBy){
		this.sortBy = new Vector (sortBy.getL().getX(),sortBy.getL().getY());
	}
	
	public int compareTo(Thing t){
		Vector x = new Vector(this.getL().getX(),this.getL().getY());
		Vector y = new Vector(t.getL().getX(),t.getL().getX());
		
		double dX = this.sortBy.distance(x);
		double dY = this.sortBy.distance(y);
		
		return (int)Math.ceil(dX-dY);
	}
	
	public Vector getCenter(){
		return Vector.add(this.loc, new Vector(this.size/2,this.size/2));
	}
	
	public Thing clone() {
		try {
			return (Thing) super.clone();
		} catch (CloneNotSupportedException e) {		
			e.printStackTrace();
			throw new RuntimeException();
		}
	}
	
	public Ellipse2D.Double getShape(){
		Ellipse2D.Double e2D = new Ellipse2D.Double(this.loc.x(),this.loc.y(),this.size,this.size);
		return e2D;
	}
}
