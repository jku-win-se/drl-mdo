/**
 */
package SprintPlanning;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Plan</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link SprintPlanning.Plan#getStakeholders <em>Stakeholders</em>}</li>
 *   <li>{@link SprintPlanning.Plan#getBacklog <em>Backlog</em>}</li>
 *   <li>{@link SprintPlanning.Plan#getSprints <em>Sprints</em>}</li>
 *   <li>{@link SprintPlanning.Plan#getMaxSprints <em>Max Sprints</em>}</li>
 *   <li>{@link SprintPlanning.Plan#getMinSprints <em>Min Sprints</em>}</li>
 *   <li>{@link SprintPlanning.Plan#getCurrentSprints <em>Current Sprints</em>}</li>
 * </ul>
 *
 * @see SprintPlanning.SprintPlanningPackage#getPlan()
 * @model
 * @generated
 */
public interface Plan extends EObject {
   /**
    * Returns the value of the '<em><b>Stakeholders</b></em>' containment reference list.
    * The list contents are of type {@link SprintPlanning.Stakeholder}.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @return the value of the '<em>Stakeholders</em>' containment reference list.
    * @see SprintPlanning.SprintPlanningPackage#getPlan_Stakeholders()
    * @model containment="true"
    * @generated
    */
   EList<Stakeholder> getStakeholders();

   /**
    * Returns the value of the '<em><b>Backlog</b></em>' containment reference.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @return the value of the '<em>Backlog</em>' containment reference.
    * @see #setBacklog(Backlog)
    * @see SprintPlanning.SprintPlanningPackage#getPlan_Backlog()
    * @model containment="true" required="true"
    * @generated
    */
   Backlog getBacklog();

   /**
    * Sets the value of the '{@link SprintPlanning.Plan#getBacklog <em>Backlog</em>}' containment reference.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @param value the new value of the '<em>Backlog</em>' containment reference.
    * @see #getBacklog()
    * @generated
    */
   void setBacklog(Backlog value);

   /**
    * Returns the value of the '<em><b>Sprints</b></em>' containment reference list.
    * The list contents are of type {@link SprintPlanning.Sprint}.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @return the value of the '<em>Sprints</em>' containment reference list.
    * @see SprintPlanning.SprintPlanningPackage#getPlan_Sprints()
    * @model containment="true"
    * @generated
    */
   EList<Sprint> getSprints();

   /**
    * Returns the value of the '<em><b>Max Sprints</b></em>' attribute.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @return the value of the '<em>Max Sprints</em>' attribute.
    * @see #setMaxSprints(int)
    * @see SprintPlanning.SprintPlanningPackage#getPlan_MaxSprints()
    * @model required="true"
    * @generated
    */
   int getMaxSprints();

   /**
    * Sets the value of the '{@link SprintPlanning.Plan#getMaxSprints <em>Max Sprints</em>}' attribute.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @param value the new value of the '<em>Max Sprints</em>' attribute.
    * @see #getMaxSprints()
    * @generated
    */
   void setMaxSprints(int value);

   /**
    * Returns the value of the '<em><b>Min Sprints</b></em>' attribute.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @return the value of the '<em>Min Sprints</em>' attribute.
    * @see #setMinSprints(int)
    * @see SprintPlanning.SprintPlanningPackage#getPlan_MinSprints()
    * @model required="true"
    * @generated
    */
   int getMinSprints();

   /**
    * Sets the value of the '{@link SprintPlanning.Plan#getMinSprints <em>Min Sprints</em>}' attribute.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @param value the new value of the '<em>Min Sprints</em>' attribute.
    * @see #getMinSprints()
    * @generated
    */
   void setMinSprints(int value);

   /**
    * Returns the value of the '<em><b>Current Sprints</b></em>' attribute.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @return the value of the '<em>Current Sprints</em>' attribute.
    * @see #setCurrentSprints(int)
    * @see SprintPlanning.SprintPlanningPackage#getPlan_CurrentSprints()
    * @model required="true"
    * @generated
    */
   int getCurrentSprints();

   /**
    * Sets the value of the '{@link SprintPlanning.Plan#getCurrentSprints <em>Current Sprints</em>}' attribute.
    * <!-- begin-user-doc -->
    * <!-- end-user-doc -->
    * @param value the new value of the '<em>Current Sprints</em>' attribute.
    * @see #getCurrentSprints()
    * @generated
    */
   void setCurrentSprints(int value);

} // Plan
