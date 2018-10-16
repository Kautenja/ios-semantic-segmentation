//
//  Popup.swift
//  Tiramisu
//
//  Created by James Kauten on 10/16/18.
//  Copyright © 2018 Kautenja. All rights reserved.
//

import Foundation
import UIKit

/// Display a popup on an input view controller with title and message.
/// Args:
///     vc: the view controller to display the popup on
///     title: the title of the popup to display
///     message: the message for the popup alert
///
func popup_alert(_ vc: ViewController, title: String, message: String) {
    // create an alert view controller with given title and message
    let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
    // create the acknowledgement action for the popup
    let alertAction = UIAlertAction(title: "OK", style: .default)
    // add the action to the popup view controller
    alert.addAction(alertAction)
    // present the popup on the input view controller
    vc.present(alert, animated: true)
}
