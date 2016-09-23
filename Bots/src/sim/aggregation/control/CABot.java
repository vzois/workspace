package sim.aggregation.control;

import java.awt.Color;

import sim.elements.Bot;
import sim.elements.Thing;
import sim.structures.Vector;

public class CABot extends Bot {
	double aF,bF,cF;
	
	public CABot(double size,double range,double aF,double bF, double cF){
		super(size,range,Color.BLUE);
		this.aF=aF;
		this.bF=bF;
		this.cF=cF;
		this.setMaxSpeed(5);
		this.setFlatWorld(true);
		//System.out.println(aF+" , "+bF+" , "+cF);
	}

	@Override
	public void act() {
		Vector v = new Vector();
		double f=0;
		//if(this.getC()==Color.BLUE){
		Thing o = null;
		for(int i = 0;i<this.getNeighbors().length;i++){
			o = (Thing)this.getNeighbors()[i];
			if(!o.equals(this))
				f=aF - bF*Math.exp(-Math.pow(this.getCenter().distance(((Thing)o).getCenter()),2)/cF);
				Vector t=new Vector(this.getCenter(),((Thing)o).getCenter());
				t.multi(f);
				v.set(v.getX() + t.getX(),v.getY() + t.getY() );
			if(v.norm()>this.getMaxSpeed()){
				v.normalize();
				v.multi(this.getMaxSpeed());
				this.setC(Color.BLUE);
			}
		}
		//if(v.norm()>1) this.velocity.set(v.getX(), v.getY());
		//else{ this.stop(); }
		if(this.velocity.contains(new Vector(0,0))) this.setC(Color.RED);
		else this.setC(Color.YELLOW);
			
		this.velocity.set(v.getX(), v.getY());
	}
}
