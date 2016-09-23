package sim.collision.avoidance.robotpair;


import java.awt.Color;

import javax.swing.Box;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sim.elements.Thing;
import sim.graphics.Window;
import sim.graphics.World;

@SuppressWarnings("serial")
public class VOWindow extends Window{
	int numB=10;
	int minSight=5,maxSight=200,numSight=100;
	int bMaxAgentNumber=10,bMinAgentNumber=1,bAgentNumber=1;
	int x=600,y=600;
	
	JPanel panel;
	JSlider sliderS;
	JLabel labelS;
	JSlider sliderB;
	JLabel labelB;
	JCheckBox cb;
	JRadioButton rb;
	
	public World createWorld() {
		World world = new World(x,y);
		world.enableDrawing(Color.CYAN, numB);

		Thing t = new Thing(numB,Color.CYAN);
		//world.addObject(t,300,340,400,500);
		t.setL(300, 300);
		BlueBot b = new BlueBot(numB,numSight);
		world.addObject(b,300,340,400,500);
		b.setL(300, 550);
		return world;
	}

	@Override
	public JPanel getSettings() {
		panel = new JPanel();
		Box settingsBox = Box.createVerticalBox();
		
		settingsBox.setAlignmentX(CENTER_ALIGNMENT);
		settingsBox.setAlignmentY(CENTER_ALIGNMENT);
		
		cb = new JCheckBox("Draw Velocity Obstacles",false);
		cb.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) {
				changeDrawVelocityObstacles(e);
			}
		});
		
		cb.setEnabled(true);
		cb.setSelected(false);
		settingsBox.add(cb);
		////////////////////////////////////////////////////
		sliderS = new JSlider();
		labelS = new JLabel("Agent's Sight Range: "+numSight);
		settingsBox.add(labelS);
		sliderS.setMinimum(minSight);
		sliderS.setMaximum(maxSight);
		sliderS.setValue(numSight);
		sliderS.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent e) {
				agentsSightChange(e);
			}
			
		});
		settingsBox.add(sliderS);
		panel.add(settingsBox);
		return panel;
	}
	
	public void agentsSightChange(ChangeEvent event){
		numSight=sliderS.getValue();
		labelS.setText("Agent's Sight Range: "+numSight);
	}
	
	public void changeDrawVelocityObstacles(ChangeEvent e){
		if(cb.isSelected()){
			World.drawVelocityObstacle = true;
		}else{
			World.drawVelocityObstacle = false;
		}
	}
}
