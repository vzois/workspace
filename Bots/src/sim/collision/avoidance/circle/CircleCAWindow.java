package sim.collision.avoidance.circle;

import java.awt.Color;
import java.util.ArrayList;

import javax.swing.Box;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import math.geom2d.Point2D;
import sim.geometry.FormationShape;
import sim.graphics.Window;
import sim.graphics.World;
import sim.structures.Vector;
import sim.velocity.obstacles.AdmissableVelocity;

@SuppressWarnings("serial")
public class CircleCAWindow extends Window{
	int x=1000,y=800;
	int pMaxAgentNumber=100,pMinAgentNumber=1,pAgentNumber=50;
	ArrayList<Point2D> targets;
	JSlider sliderP;
	JLabel labelP;
	JCheckBox cbShape = new JCheckBox("Change Shape",true);
	JCheckBox cbRandom = new JCheckBox("Random Targets",true);
	JCheckBox cbPruneSearchSpace = new JCheckBox("Prune Velocities",false);
	
	@Override
	public World createWorld() {
		FormationShape fs = new FormationShape(pAgentNumber);
		if(cbShape.isSelected())fs.createCircle(new Vector(x/2,y/2), 380);
		else fs.createSquare(new Vector(x/2,y/2), 700);
		targets = new ArrayList<Point2D>(fs.getShape());
		World world = new World(x,y);
		for(int i=0;i<pAgentNumber;i++){
			CircleBot cb = new CircleBot(new Color(world.seed.nextFloat(),world.seed.nextFloat(),world.seed.nextFloat()));
			world.addObject(cb);
			Vector start = new Vector(fs.getShape().get(i));
			Vector end = new Vector(targets.get(world.seed.nextInt(targets.size())));
			if(cbRandom.isSelected()){
				while(start.equals(end)) end = new Vector(targets.get(world.seed.nextInt(targets.size())));
			}else{
				for(Point2D t : fs.getShape()){
					if(start.distance(end)< start.distance(t) && !start.equals(t)){
						end = new Vector(t);
					}
				}
			}
			cb.setL(start);
			cb.setTarget(end);
			targets.remove(end);
		}
		AdmissableVelocity.prune_search_space = this.cbPruneSearchSpace.isSelected();
		return world;
	}

	@Override
	public JPanel getSettings() {
		JPanel panel = new JPanel();
		Box settingsBox = Box.createVerticalBox();
		settingsBox.setAlignmentX(CENTER_ALIGNMENT);
		settingsBox.setAlignmentY(CENTER_ALIGNMENT);

		///////////////////////////////////////////////////////////////////
		sliderP = new JSlider();
		labelP = new JLabel("Number of Agents: "+this.pAgentNumber);
		settingsBox.add(labelP);
		sliderP.setMinimum(this.pMinAgentNumber);
		sliderP.setMaximum(this.pMaxAgentNumber);
		sliderP.setValue(this.pAgentNumber);
		sliderP.addChangeListener(new ChangeListener()
		{
			public void stateChanged(final ChangeEvent event)
			{
				agentsChangeP(event);
			}
		});		
		settingsBox.add(sliderP);
		
		settingsBox.add(cbShape);
		settingsBox.add(cbRandom);
		settingsBox.add(cbPruneSearchSpace);
		panel.add(settingsBox);
		return panel;
	}
	
	public void agentsChangeP(ChangeEvent event){
		this.pAgentNumber=sliderP.getValue();
		labelP.setText("Purple Agent Number: "+this.pAgentNumber);
	}
}
